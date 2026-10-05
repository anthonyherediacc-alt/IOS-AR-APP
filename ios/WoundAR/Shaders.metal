#include <metal_stdlib>
using namespace metal;

// Camera background + wound compositing. The camera pass follows Apple's ARKit Metal template ("Displaying an AR
// experience with Metal": Y and CbCr plane textures, YCbCr → RGB matrix). The wound is drawn in camera-image space
// from the SAME frame, so it can never lag behind the hand on screen.

constant float4x4 ycbcrToRGB = float4x4(
    float4(+1.0000f, +1.0000f, +1.0000f, +0.0000f),
    float4(+0.0000f, -0.3441f, +1.7720f, +0.0000f),
    float4(+1.4020f, -0.7141f, +0.0000f, +0.0000f),
    float4(-0.7010f, +0.5291f, -0.8860f, +1.0000f));

static float3 cameraRGB(texture2d<float> texY, texture2d<float> texCbCr, sampler s, float2 uv, float videoRange) {
    float y = texY.sample(s, uv).r;
    float2 c = texCbCr.sample(s, uv).rg;
    if (videoRange > 0.5) {
        y = (y - 16.0 / 255.0) * (255.0 / 219.0);
        c = (c - 128.0 / 255.0) * (255.0 / 224.0) + 128.0 / 255.0;
    }
    return saturate((ycbcrToRGB * float4(y, c, 1.0)).rgb);
}

static float luma(float3 c) { return dot(c, float3(0.299, 0.587, 0.114)); }

// MARK: Camera background

struct CameraOut {
    float4 position [[position]];
    float2 image;
};

// corners[i] = (clip x, clip y, image u, image v), triangle-strip order; image coordinates come from the inverse of
// ARFrame.displayTransform on the CPU (aspect fill, portrait).
vertex CameraOut cameraVertex(uint vid [[vertex_id]], constant float4 *corners [[buffer(0)]]) {
    CameraOut out;
    out.position = float4(corners[vid].xy, 0.0, 1.0);
    out.image = corners[vid].zw;
    return out;
}

fragment float4 cameraFragment(CameraOut in [[stage_in]],
                               texture2d<float> texY [[texture(0)]],
                               texture2d<float> texCbCr [[texture(1)]],
                               constant float4 &params [[buffer(0)]]) {
    constexpr sampler s(filter::linear, address::clamp_to_edge);
    return float4(cameraRGB(texY, texCbCr, s, in.image, params.x), 1.0);
}

// MARK: Wound

struct WoundVertex {
    float4 imageUV; // camera-image position (normalized 0…1), wound texture uv
    float4 extra;   // x: visibility from depth (0…1)
};

struct WoundUniforms {
    float4 display;  // ARFrame.displayTransform a, b, c, d (normalized image → normalized view)
    float4 displayT; // tx, ty
    float4 params;   // opacity, skin reference luma, skin blending on/off, occluder capsule count
    float4 texel;    // 1 / image width, 1 / image height, debug tint, video range
};

struct Capsule {
    float4 ends;   // a.xy, b.xy in camera-image pixels
    float4 params; // radius, edge softness (pixels)
};

struct WoundOut {
    float4 position [[position]];
    float2 image;
    float2 uv;
    float visibility;
};

vertex WoundOut woundVertex(uint vid [[vertex_id]],
                            const device WoundVertex *vertices [[buffer(0)]],
                            constant WoundUniforms &u [[buffer(1)]]) {
    WoundVertex v = vertices[vid];
    float2 p = v.imageUV.xy;
    float2 view = float2(u.display.x * p.x + u.display.z * p.y + u.displayT.x,
                         u.display.y * p.x + u.display.w * p.y + u.displayT.y);
    WoundOut out;
    out.position = float4(view.x * 2.0 - 1.0, 1.0 - view.y * 2.0, 0.0, 1.0);
    out.image = p;
    out.uv = v.imageUV.zw;
    out.visibility = v.extra.x;
    return out;
}

static float capsuleDistance(float2 p, float2 a, float2 b) {
    float2 pa = p - a, ba = b - a;
    float h = clamp(dot(pa, ba) / max(dot(ba, ba), 1e-6), 0.0, 1.0);
    return length(pa - ba * h);
}

fragment float4 woundFragment(WoundOut in [[stage_in]],
                              texture2d<float> texY [[texture(0)]],
                              texture2d<float> texCbCr [[texture(1)]],
                              texture2d<float> wound [[texture(2)]],
                              constant WoundUniforms &u [[buffer(0)]],
                              constant Capsule *capsules [[buffer(1)]]) {
    constexpr sampler s(filter::linear, address::clamp_to_edge);
    float videoRange = u.texel.w;
    float3 skin = cameraRGB(texY, texCbCr, s, in.image, videoRange);

    // Wound picture (premultiplied RGBA) → straight colour.
    float4 w = wound.sample(s, in.uv);
    float3 woundRGB = w.rgb / max(w.a, 1e-4);

    // Never show the picture's rectangle: fade to nothing at its border.
    float2 e = min(in.uv, 1.0 - in.uv);
    float border = smoothstep(0.0, 0.08, min(e.x, e.y));

    // Other hand in front: soft capsules around its finger bones and palm (camera-image pixels).
    float2 px = in.image / u.texel.xy;
    float cover = 0.0;
    int count = int(u.params.w);
    for (int i = 0; i < count; i++) {
        float d = capsuleDistance(px, capsules[i].ends.xy, capsules[i].ends.zw);
        cover = max(cover, 1.0 - smoothstep(capsules[i].params.x - capsules[i].params.y, capsules[i].params.x, d));
    }

    float visible = saturate(in.visibility) * (1.0 - cover) * border * u.params.x;
    float3 result;
    if (u.params.z > 0.5) {
        // Skin-aware compositing (shading/"ratio" transfer as in Bradley & Roth 2004 / Pilet et al. 2008):
        // the skin's own low-frequency shading under the wound modulates it, so it darkens where the hand does.
        float2 t = u.texel.xy * 7.0;
        float3 blur = (cameraRGB(texY, texCbCr, s, in.image + float2(t.x, 0), videoRange)
                     + cameraRGB(texY, texCbCr, s, in.image - float2(t.x, 0), videoRange)
                     + cameraRGB(texY, texCbCr, s, in.image + float2(0, t.y), videoRange)
                     + cameraRGB(texY, texCbCr, s, in.image - float2(0, t.y), videoRange)
                     + skin) / 5.0;
        float shade = clamp(luma(blur) / max(u.params.y, 0.05), 0.35, 1.5);
        // The deep cut (opaque core) replaces skin; the halo only reddens the skin, so pores, hair and lighting stay.
        float core = smoothstep(0.75, 0.95, w.a);
        float3 hue = woundRGB / max(max(woundRGB.r, max(woundRGB.g, woundRGB.b)), 0.05);
        float3 halo = skin * mix(float3(1.0), hue, w.a * 0.55);
        float3 composite = mix(halo, woundRGB * shade, core);
        result = mix(skin, composite, visible);
    } else {
        result = mix(skin, woundRGB, w.a * visible);
    }
    // Debug: tint the whole mesh so its extent is visible.
    result = mix(result, float3(0.0, 1.0, 0.4), u.texel.z * 0.25);
    return float4(result, 1.0);
}
