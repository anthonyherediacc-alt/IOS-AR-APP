/*! thin-plate-spline 0.1.0 — https://github.com/pravoobi/try-on (npm: thin-plate-spline)
 * MIT License
 * 
 * Copyright (c) 2026 pravoobi
 * 
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 * 
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 * 
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */
/**
 * Thin-plate-spline (TPS) image warping — solve a smooth, nonlinear
 * point-correspondence warp from an arbitrary set of source/destination
 * control points, then render it to a canvas as a coarse grid of
 * affine-textured triangles (canvas 2D can only do affine transforms, so a
 * full nonlinear TPS warp is approximated locally per triangle).
 *
 * No dependencies beyond a canvas-like 2D context (OffscreenCanvas or a
 * regular <canvas>) and TypeScript itself — usable in a Worker or the main
 * thread, Node or the browser (given a canvas polyfill).
 */
// ---------------------------------------------------------------------------
// Thin plate spline (pure math, no deps)
// ---------------------------------------------------------------------------
/** Radial basis kernel for TPS. U(0) = 0 by convention. */
function radialBasis(r2) {
    return r2 <= 0 ? 0 : r2 * Math.log(r2);
}
/** Solves the linear system Ax = b via Gaussian elimination with partial pivoting. */
function solveLinearSystem(a, b) {
    const n = b.length;
    const m = a.map((row) => row.slice());
    const rhs = b.slice();
    for (let col = 0; col < n; col++) {
        let pivot = col;
        for (let row = col + 1; row < n; row++) {
            if (Math.abs(m[row][col]) > Math.abs(m[pivot][col]))
                pivot = row;
        }
        if (Math.abs(m[pivot][col]) < 1e-12) {
            throw new Error('tps-warp: singular TPS system (control points may be collinear/duplicated)');
        }
        if (pivot !== col) {
            [m[col], m[pivot]] = [m[pivot], m[col]];
            [rhs[col], rhs[pivot]] = [rhs[pivot], rhs[col]];
        }
        for (let row = col + 1; row < n; row++) {
            const factor = m[row][col] / m[col][col];
            if (factor === 0)
                continue;
            for (let k = col; k < n; k++)
                m[row][k] -= factor * m[col][k];
            rhs[row] -= factor * rhs[col];
        }
    }
    const x = new Array(n).fill(0);
    for (let row = n - 1; row >= 0; row--) {
        let sum = rhs[row];
        for (let k = row + 1; k < n; k++)
            sum -= m[row][k] * x[k];
        x[row] = sum / m[row][row];
    }
    return x;
}
/**
 * Interpolates src -> dst through n control point correspondences (n >= 3,
 * non-collinear). If dst_i = A(src_i) for some affine map A, the fit reduces
 * to exactly A everywhere (the TPS "affine reduction" property) — this is
 * what the unit tests exercise.
 */
export class ThinPlateSpline {
    controlPoints;
    wx;
    wy;
    ax;
    ay;
    constructor(src, dst) {
        if (src.length !== dst.length)
            throw new Error('tps-warp: src/dst length mismatch');
        if (src.length < 3)
            throw new Error('tps-warp: need at least 3 control points');
        const n = src.length;
        this.controlPoints = src.slice();
        // Build the (n+3)x(n+3) system [[K, P], [P^T, 0]] * [w; a] = [v; 0].
        const size = n + 3;
        const matrix = Array.from({ length: size }, () => new Array(size).fill(0));
        for (let i = 0; i < n; i++) {
            for (let j = 0; j < n; j++) {
                const dx = src[i][0] - src[j][0];
                const dy = src[i][1] - src[j][1];
                matrix[i][j] = radialBasis(dx * dx + dy * dy);
            }
            matrix[i][n] = 1;
            matrix[i][n + 1] = src[i][0];
            matrix[i][n + 2] = src[i][1];
            matrix[n][i] = 1;
            matrix[n + 1][i] = src[i][0];
            matrix[n + 2][i] = src[i][1];
        }
        const rhsX = [...dst.map((p) => p[0]), 0, 0, 0];
        const rhsY = [...dst.map((p) => p[1]), 0, 0, 0];
        const solX = solveLinearSystem(matrix, rhsX);
        const solY = solveLinearSystem(matrix, rhsY);
        this.wx = solX.slice(0, n);
        this.ax = [solX[n], solX[n + 1], solX[n + 2]];
        this.wy = solY.slice(0, n);
        this.ay = [solY[n], solY[n + 1], solY[n + 2]];
    }
    eval(p) {
        let x = this.ax[0] + this.ax[1] * p[0] + this.ax[2] * p[1];
        let y = this.ay[0] + this.ay[1] * p[0] + this.ay[2] * p[1];
        for (let i = 0; i < this.controlPoints.length; i++) {
            const dx = p[0] - this.controlPoints[i][0];
            const dy = p[1] - this.controlPoints[i][1];
            const u = radialBasis(dx * dx + dy * dy);
            x += this.wx[i] * u;
            y += this.wy[i] * u;
        }
        return [x, y];
    }
}
export const DEFAULT_WARP_GRID = { cols: 16, rows: 24 };
/**
 * Draws `image` into the destination triangle (dx0,dy0)-(dx1,dy1)-(dx2,dy2),
 * sampling from the source triangle (sx0,sy0)-(sx1,sy1)-(sx2,sy2) in image
 * pixel space. Standard canvas affine-texture-mapping trick: solve for the
 * transform that satisfies all 3 correspondences, clip to the destination
 * triangle, then draw the whole image through that transform.
 */
function drawTexturedTriangle(ctx, image, src, dst) {
    const [[sx0, sy0], [sx1, sy1], [sx2, sy2]] = src;
    const [[dx0, dy0], [dx1, dy1], [dx2, dy2]] = dst;
    const u1 = sx1 - sx0;
    const v1 = sy1 - sy0;
    const u2 = sx2 - sx0;
    const v2 = sy2 - sy0;
    const x1 = dx1 - dx0;
    const y1 = dy1 - dy0;
    const x2 = dx2 - dx0;
    const y2 = dy2 - dy0;
    const delta = u1 * v2 - u2 * v1;
    if (Math.abs(delta) < 1e-9)
        return; // degenerate triangle (zero area in source)
    const a = (x1 * v2 - x2 * v1) / delta;
    const b = (y1 * v2 - y2 * v1) / delta;
    const c = (x2 * u1 - x1 * u2) / delta;
    const d = (y2 * u1 - y1 * u2) / delta;
    const e = dx0 - a * sx0 - c * sy0;
    const f = dy0 - b * sx0 - d * sy0;
    ctx.save();
    ctx.beginPath();
    ctx.moveTo(dx0, dy0);
    ctx.lineTo(dx1, dy1);
    ctx.lineTo(dx2, dy2);
    ctx.closePath();
    ctx.clip();
    ctx.setTransform(a, b, c, d, e, f);
    ctx.drawImage(image, 0, 0);
    ctx.restore();
}
/**
 * Bounding box of a set of points, expanded by a fraction of its own
 * width/height, clamped to [0, maxW] x [0, maxH].
 */
function expandedBBox(points, marginXFrac, marginYFrac, maxW, maxH) {
    const xs = points.map((p) => p[0]);
    const ys = points.map((p) => p[1]);
    const minX = Math.min(...xs);
    const maxX = Math.max(...xs);
    const minY = Math.min(...ys);
    const maxY = Math.max(...ys);
    const mx = (maxX - minX) * marginXFrac;
    const my = (maxY - minY) * marginYFrac;
    return {
        minX: Math.max(0, minX - mx),
        maxX: Math.min(maxW, maxX + mx),
        minY: Math.max(0, minY - my),
        maxY: Math.min(maxH, maxY + my),
    };
}
/**
 * Warps `image` onto an output canvas of (outputWidth x outputHeight), per
 * the TPS mapping from `srcAnchors` to `dstAnchors` (arbitrary point
 * correspondences — no fixed anchor count or naming beyond "at least 3,
 * non-collinear"). Output is transparent outside the mapped region.
 *
 * `marginXFrac`/`marginYFrac` (default 0.6/0.15) control how far past the
 * anchors' own bounding box the sampled grid extends, since content like
 * sleeves can extend past the anchors laterally — but pixels far outside
 * the control-point convex hull are where TPS extrapolation gets unstable
 * and can fold, so keep this modest rather than sampling the whole image.
 */
export function renderGarmentWarp(image, srcAnchors, dstAnchors, outputWidth, outputHeight, grid = DEFAULT_WARP_GRID, marginXFrac = 0.6, marginYFrac = 0.15) {
    const tps = new ThinPlateSpline(srcAnchors, dstAnchors);
    const canvas = new OffscreenCanvas(outputWidth, outputHeight);
    const ctx = canvas.getContext('2d');
    if (!ctx)
        throw new Error('tps-warp: no 2d context');
    const { cols, rows } = grid;
    const imgW = image.width;
    const imgH = image.height;
    const bbox = expandedBBox(srcAnchors, marginXFrac, marginYFrac, imgW, imgH);
    // Precompute mapped grid vertices: srcGrid in source-image space, dstGrid
    // warped into output space via the TPS.
    const srcGrid = [];
    const dstGrid = [];
    for (let j = 0; j <= rows; j++) {
        const srcRow = [];
        const dstRow = [];
        const v = bbox.minY + (bbox.maxY - bbox.minY) * (j / rows);
        for (let i = 0; i <= cols; i++) {
            const u = bbox.minX + (bbox.maxX - bbox.minX) * (i / cols);
            const p = [u, v];
            srcRow.push(p);
            dstRow.push(tps.eval(p));
        }
        srcGrid.push(srcRow);
        dstGrid.push(dstRow);
    }
    for (let j = 0; j < rows; j++) {
        for (let i = 0; i < cols; i++) {
            const s00 = srcGrid[j][i];
            const s10 = srcGrid[j][i + 1];
            const s01 = srcGrid[j + 1][i];
            const s11 = srcGrid[j + 1][i + 1];
            const d00 = dstGrid[j][i];
            const d10 = dstGrid[j][i + 1];
            const d01 = dstGrid[j + 1][i];
            const d11 = dstGrid[j + 1][i + 1];
            drawTexturedTriangle(ctx, image, [s00, s10, s01], [d00, d10, d01]);
            drawTexturedTriangle(ctx, image, [s10, s11, s01], [d10, d11, d01]);
        }
    }
    return canvas;
}
