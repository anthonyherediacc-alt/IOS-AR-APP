# Experimental body / mannequin sensing

## Scope and baseline

Baseline: `f742507b9654dc71310571efb40bb7ea40ef058a`, default branch
`claude/safari-ar-wound-sim-6nb3e1`. No repository AGENTS.md or committed tests were found.
The browser is a static MediaPipe/Three.js app; the native app is SwiftUI + one ARSession,
Vision hand pose, skin optical flow, and Metal. Existing historical measurements in PROJECT.md
are not measurements from this change. Browser files, skin tracker, Metal renderer/shaders,
hand math and third-party dependencies are unchanged.

The iPhone supplies sensing and diagnostics. Unity owns the patient asset, animation, wounds,
clinical interactions and scoring. The new ghost is a skeleton / inferred capsule envelope,
not a scan. No mannequin dimensions or teammate asset properties are assumed.

## Run the experiment

1. Build using the existing XcodeGen workflow and sideload the unsigned IPA as in README.md.
2. Check the original Hand Wound Demo before and after switching modes. Its original switches remain available.
3. Select Body / Mannequin or Depth / Diagnostics. Select Vision 2D + depth first.
4. Inspect requested/active capabilities. Non-LiDAR devices can produce 2D-only observations.
5. Frame a consenting person, then the actual CRiSis Adult ALS Manikin LF03890IP lying supine.
   Tap a candidate center to select it. The human/mannequin label is operator metadata.
6. Compare raw landmarks, projected 3D surface samples and filtered samples. Increase the
   smoothing time constant only while explicitly evaluating its lag. Depth and confidence
   visualizations use the same processed frame. The envelope is inferred and view-space.
7. Repeat with the separately selected ARKit body backend if supported. Never interpret its
   inferred joint flag as a probability or a directly measured anatomical joint.

The display uses a portrait, aspect-fit processed image and its matching camera. It does not
put old landmarks over a newer live camera. A TimelineView labels image age independently of
inference; automatic joint graphics disappear at 0.75 seconds of age. Frozen manual-placement
images are prominently labeled. This display policy trades latency for synchronization.

## Session and identity policy

WoundSession remains the sole ARSession owner. Mode/backend changes, backgrounding, interruption
and reset pause capture, increment the world-origin epoch and invalidate visible results,
manual calibration and outstanding inference publication. Serial inference permits one frame
in flight; frames arriving while busy are discarded. The writer has a separate bounded queue.
The original hand pipeline is reset on mode entry; its algorithm is not used for body tracking.

Vision runs on a clockwise-rotated portrait image. Its bottom-left normalized result `(x,y)`
maps to raw sensor top-left UV `(1-y,1-x)`. Display UV is `(1-v,u)` with no crop. Depth dimensions,
row strides, camera intrinsics and transforms come from that same ARFrame. Unprojection uses
`((u*width-cx)*d/fx, -(v*height-cy)*d/fy, -d)` in the ARKit camera frame.

Vision identity uses a conservative image-centroid gate (0.12 normalized units). Multiple nearby
candidates, a missing match or a gap over 0.75 seconds requires explicit reselection. ARKit
also locks the anchor identifier. These are experimental heuristics, not robust visual identity;
an undetected person replacing a mannequin in the same location can still fool the matcher.
Do not claim zero identity swaps. The implementation does not predict through loss.

The world configuration requests raw and smoothed scene depth only when each is supported.
Body estimation deliberately samples raw sceneDepth; smoothed depth remains available to the
preserved hand path. ARBodyTrackingConfiguration is separate, with no depth or reconstruction
requested, and never consumes a depth map from the previous configuration.

Depth uses high-confidence 3x3 neighborhoods with at least five finite positive values and a
discontinuity gate. These gates are engineering defaults, not validated clinical thresholds.
Depth measures a visible surface, not a joint center. A uniformly sampled trainee/floor can
pass this gate; there is no semantic depth segmentation. Fields and UI keep that limitation visible.
Chest/pelvis are derived midpoints only when both input landmarks exist. The optional Vision
root uses shoulder direction, torso direction and their cross product, not gravity; its normal
is geometric and is not a measured skin normal. Missing geometry produces no root pose.

## Assisted calibration and attachments

Freeze an image, choose a named landmark, and tap its anatomical location. Placement requires
reliable same-frame depth; there is no fallback raycast onto the floor. Verify the surface
visually. Records include source, original frame/time/epoch and `staticAssistedUnverified` status.

Supply a JSON object mapping the same names to **actual supplied reference coordinates in metres**:
each value is an array of three finite numbers. At least three noncollinear, distinct matching
landmarks are required; use more well-spaced torso points and reserve independent held-out points.
The Horn/Jacobi rigid fit allows planar sets and preserves proper rotations. It reports RMS fit
residual and measured/reference scale ratio without applying scale. A ratio far from one needs
investigation of units, correspondence and depth; fit residual is not true alignment accuracy.

Static calibration axes are dashed; manual points are gray. This is static assisted alignment,
not automatic articulated tracking. Explicitly invalidate it when the mannequin or a limb moves.
Observed disagreement over 10 cm also invalidates it (experimental gate). With no automatic
observation, motion cannot be detected, so static alignment remains operator-dependent.

The pink marker refers to a named live landmark. Export includes its world position and, when
a root exists, body-local position. It becomes unavailable on target loss. It is a diagnostic
anatomical reference, not a validated skin attachment: a final asset needs surface correspondence,
normals, offsets and independent registration validation. Screen selection does not measure
contact, pressure or correct treatment.

## Recording and privacy

Start/stop recording explicitly. The red indicator denotes pose recording. Pose-only is the
default. An additional opt-in saves portrait JPEGs, raw sensor depth and confidence at at most
1 fps. All inference/storage is local. Stop before export. Delete the current recording in the
app; older session folders can be exported/deleted through Files > On My iPhone > Wound AR.
Session-folder sharing depends on the chosen share destination; Files is the fallback.

Each session folder contains `pose.jsonl`. Opted-in imagery uses frame-ID filenames:
`N.jpg`, `N.depth`, `N.confidence`, `N.json`. JPEG pixels are physically normalized to portrait.
Depth is tightly packed little-endian Float32 metres in raw sensor orientation; confidence is
UInt8 0/1/2 with the same dimensions. Invalid raw depth remains invalid in binary and must be
rejected by consumers. Metadata records dimensions, intrinsics, camera pose, epoch and timestamp.
No fabricated depth is saved when unavailable. Captured RGB is not a recoverable ARKit session.

Each recording is capped at 50 MiB pose plus 100 MiB imagery; new recording is refused once
existing storage is at least 250 MiB (bounded aggregate below 400 MiB). A slow writer drops
frames rather than retaining unlimited snapshots; timestamp/frame-ID gaps are expected.
No recordings, signing materials, contact details or generated binaries belong in Git.

## Version 1 export contract

One JSON object per line, defined by TrackingPacket in TrackingCore/TrackingTypes.swift:

| Field | Meaning |
| --- | --- |
| schemaVersion | 1; reject unsupported versions |
| sessionID, frameID | Pipeline session UUID and increasing processed-frame ID; skipped IDs are possible |
| timestamp | ARFrame monotonic seconds since device startup, not UTC |
| worldOriginEpoch | Changes on session reset/configuration/interruption; never bridge epochs implicitly |
| backend, targetID, targetLabel | Detector, operator-selected track UUID, operator label |
| state, observationAge | searching/acquiring/tracking/temporarilyLost/lost; age since accepted observation |
| cameraPose, bodyRootPose | Column-major 4x4 camera-to-world and optional root-to-world matrices |
| joints | Named optional image/world/filteredWorld positions with provenance, confidence, tracked flag and missing-data reason |
| calibrationID, calibration, manualLandmarks | Explicit static assisted registration and source records |
| attachment | Named diagnostic reference; optional world/bodyLocal positions; not physical contact |
| events | Selection, identity/state discontinuities and invalidation reasons |
| inferenceMS, smoothingTimeConstant | Inference processing time and configured filter time constant |

Positions are metres in ARKit right-handed world coordinates (+Y up, initial camera looks -Z).
Image positions are raw sensor top-left normalized UV. Missing fields are omitted, never replaced
with `(0,0,0)`; JSONEncoder rejects NaN/infinity. No quaternion fields are emitted in v1. If a
consumer derives one, use `(x,y,z,w)` and document the coordinate basis. Vision confidence is
detector confidence; ARKit `tracked` is a Boolean. Neither indicates depth probability.

Vision names preserve anatomical left/right and use camelCase (leftShoulder, leftElbow,
leftWrist, leftHip, leftKnee, leftAnkle and right equivalents; nose, neck; derived chest/pelvis).
ARKit preserves the SDK's complete joint names and parentName hierarchy; no nonexistent joints
are fabricated. Consumers must map those exact names to their rig deliberately.

## Unity replay and shared-space alignment

Copy integration/examples/PoseReplay.cs into a disposable Unity project; attach it to an empty
GameObject. Rename an exported JSONL to `.txt`, import it as a TextAsset, and assign it. Enable
Gizmos to see joints/segments. Playback follows recorded timestamps, suppresses lost/stale data,
and does not interpolate across gaps. It is a debug sample, not a retargeting system.
Unity Editor/package version actually tested: **none yet**. The sample uses UnityEngine only,
targets Unity versions supporting Time.realtimeSinceStartupAsDouble, and requires no AR package.

Basis conversion uses `B=diag(1,1,-1,1)`, positions `B*p`, rotations/poses `B*T*B`.
Negating a position's Z alone does not convert a rotation. Swift tests cover a known 90-degree
rotation and a matrix round trip. Anatomical left/right never changes with this basis conversion.
Joint positions alone cannot resolve bone twist or fully drive an arbitrary custom rig.

The sharedSpace Transform is a **separate** estimated transform from converted phone coordinates
to the consumer's scene. Identity is only a replay convention. For a headset, measure common
correspondences, validate on held-out points, estimate clock offset/latency, and invalidate on
either device's origin reset. A coordinate basis change does not align independent devices.

Future paths: move selected sensing into a Unity iOS app through AR Foundation/ARKit, or retain
the phone as an external sensor with an explicit transport/shared-space/time protocol. ARKit in
Unity does not create that bridge. Live streaming and marker assistance are intentionally deferred.

## Reproducible evaluation

On macOS: `cd ios && swift test`; `swiftc -typecheck ../tools/infer-images.swift`.
Anywhere with Node: `node --test tools/evaluate.test.mjs`.
Offline consented images: `swift tools/infer-images.swift image1.jpg image2.png > detections.jsonl`.
Extract sampled video frames with their timestamps using a local video tool, retaining EXIF
orientation; run the same image command. This reruns Vision on RGB and cannot restore LiDAR.
The offline tool reports oriented-image top-left UV, unlike the live export's raw sensor UV;
convert annotations explicitly. It preserves Vision's native joint keys.

Pose evaluation: `node tools/evaluate.mjs pose.jsonl annotations.jsonl`.
Each annotation has sessionID, worldOriginEpoch, frameID, targetPresent (Boolean), and a joints
object mapping names to independent raw sensor normalized UV. For a portrait captured JPEG,
convert annotated `(x,y)` to sensor `(y,1-x)`. The evaluator rejects missing frames, reports
selected-target availability, negative false positives and mean matched landmark UV error,
and leaves unavailable measurements null. It does not automatically infer identity swaps,
detector recall before selection, metric accuracy or annotation validity.

Run repeated trials on upright humans and supine/seated mannequins, with floor/table/stretcher,
partial framing, lighting changes, phone movement, rigid target movement, limb repositioning,
trainee occlusion, interruptions, empty room and trainee-only negatives. Annotate actual target
identity independently. Record acquisition time/success, detector availability, identity swaps,
landmark error, held-out registration error, stationary jitter, dropout duration/recovery,
inference time, displayed age, memory and thermal behavior. Use pixels/UV unless a credible
metric reference supports millimetres/degrees. No performance thresholds have been approved.

First device experiment: supine mannequin, phone stationary, then phone orbit, then move one
arm, then have a trainee occlude torso. Run both backends independently; capture negatives too.
Check whether the observed anatomical landmarks belong to the mannequin, not merely whether
the detector returned a body. Keep a signed/consented data record outside the repository.

## Acceptance status and remaining inputs

Milestones A–C have experimental implementation and test infrastructure, **not completed physical
acceptance**. Automatic mannequin recognition is incomplete until demonstrated on actual imagery.
No supplied mannequin footage, labeled dataset, reference landmark dimensions, Unity asset/project
or iPhone was available here. No mannequin-specific model weights were selected or trained.
BodyDetector is replaceable; evaluate the Apple baseline before justifying an alternate licensed
pose estimator. A bounding-box-only model cannot drive the required articulated landmarks.

Optional Vision 3D remains deferred until the baseline is device-validated. Its AVDepthData input
is not interchangeable with ARDepthData; scale/root conventions and runtime cost require a
separate experiment. Reconstruction/point-cloud rendering, marker initialization, metric benchmark
automation and live transport remain future work. This PR makes no clinical-performance claim.

## Technical references

- [Apple configuration-specific semantics](https://developer.apple.com/documentation/arkit/arconfiguration/supportsframesemantics(_:))
- [Apple scene depth](https://developer.apple.com/documentation/arkit/arconfiguration/framesemantics-swift.struct/scenedepth)
- [Apple skeleton model/local transforms](https://developer.apple.com/documentation/arkit/arskeleton3d)
- [Apple joint tracking flags](https://developer.apple.com/documentation/arkit/arskeleton/isjointtracked(_:))
- [Apple skeleton joint hierarchy](https://developer.apple.com/documentation/arkit/validating-a-model-for-motion-capture)
- [Apple Vision 3D and AVDepthData](https://developer.apple.com/documentation/vision/vndetecthumanbodypose3drequest)
- [Unity JSON serialization](https://docs.unity3d.com/6000.0/Documentation/ScriptReference/JsonUtility.FromJson.html)
