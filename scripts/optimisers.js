const math = window.math;

if (!math) {
    throw new Error("math.js is required for optimisers.");
}

export const DEFAULT_NUM_STEPS = 20;
const EPSILON = 1e-9;
const ADAM_EPSILON = 1e-8;

function evaluateNumber(compiled, scope) {
    try {
        const value = compiled.evaluate(scope);
        const numeric = typeof value === "number" ? value : Number(value);
        return Number.isFinite(numeric) ? numeric : NaN;
    } catch (error) {
        return NaN;
    }
}

function ensureValue(value, fallback) {
    return Number.isFinite(value) ? value : fallback;
}

function invert2x2(a, b, c, d) {
    const det = a * d - b * c;
    if (!Number.isFinite(det) || Math.abs(det) < EPSILON) {
        return null;
    }
    const invDet = 1 / det;
    return {
        xx: d * invDet,
        xy: -b * invDet,
        yx: -c * invDet,
        yy: a * invDet,
    };
}

export function analyzeFunction1D(functionStr) {
    try {
        const expr = math.parse(functionStr);
        const compiled = expr.compile();
        const gradientExpr = math.derivative(expr, "x");
        const gradientCompiled = gradientExpr.compile();
        const hessianExpr = math.derivative(gradientExpr, "x");
        const hessianCompiled = hessianExpr.compile();

        return {
            functionStr,
            fn: compiled,
            functionNode: expr,
            gradientNode: gradientExpr,
            gradient: gradientCompiled,
            hessianNode: hessianExpr,
            hessian: hessianCompiled,
        };
    } catch (error) {
        throw new Error(`Unable to parse univariate function: ${error.message}`);
    }
}

export function analyzeFunction2D(functionStr) {
    try {
        const expr = math.parse(functionStr);
        const compiled = expr.compile();

        const gradXNode = math.derivative(expr, "x");
        const gradYNode = math.derivative(expr, "y");
        const gradXCompiled = gradXNode.compile();
        const gradYCompiled = gradYNode.compile();

        const hessXXNode = math.derivative(gradXNode, "x");
        const hessXYNode = math.derivative(gradXNode, "y");
        const hessYXNode = math.derivative(gradYNode, "x");
        const hessYYNode = math.derivative(gradYNode, "y");

        return {
            functionStr,
            fn: compiled,
            functionNode: expr,
            gradientNodes: {
                x: gradXNode,
                y: gradYNode,
            },
            gradient: {
                x: gradXCompiled,
                y: gradYCompiled,
            },
            hessianNodes: {
                xx: hessXXNode,
                xy: hessXYNode,
                yx: hessYXNode,
                yy: hessYYNode,
            },
            hessian: {
                xx: hessXXNode.compile(),
                xy: hessXYNode.compile(),
                yx: hessYXNode.compile(),
                yy: hessYYNode.compile(),
            },
        };
    } catch (error) {
        throw new Error(`Unable to parse bivariate function: ${error.message}`);
    }
}

export function computeTrajectory1D(analysis, settings) {
    const numSteps = settings.numSteps ?? DEFAULT_NUM_STEPS;
    const xs = new Array(numSteps + 1).fill(0);
    const ys = new Array(numSteps + 1).fill(0);

    xs[0] = settings.initialX;
    ys[0] = ensureValue(evaluateNumber(analysis.fn, { x: xs[0] }), 0);

    if (settings.optimiserType === "Gradient Descent") {
        for (let i = 0; i < numSteps; i += 1) {
            const gradVal = evaluateNumber(analysis.gradient, { x: xs[i] });
            if (!Number.isFinite(gradVal)) {
                xs[i + 1] = xs[i];
                ys[i + 1] = ys[i];
                continue;
            }

            const momentumTerm = i === 0 ? 0 : settings.momentum * (xs[i] - xs[i - 1]);
            xs[i + 1] = xs[i] - settings.learningRate * gradVal + momentumTerm;
            ys[i + 1] = ensureValue(evaluateNumber(analysis.fn, { x: xs[i + 1] }), ys[i]);
        }
    } else if (settings.optimiserType === "Newton") {
        for (let i = 0; i < numSteps; i += 1) {
            const gradVal = evaluateNumber(analysis.gradient, { x: xs[i] });
            const hessVal = evaluateNumber(analysis.hessian, { x: xs[i] });
            if (!Number.isFinite(gradVal) || !Number.isFinite(hessVal) || Math.abs(hessVal) < EPSILON) {
                xs[i + 1] = xs[i];
                ys[i + 1] = ys[i];
                continue;
            }

            xs[i + 1] = xs[i] - gradVal / hessVal;
            ys[i + 1] = ensureValue(evaluateNumber(analysis.fn, { x: xs[i + 1] }), ys[i]);
        }
    } else {
        throw new Error(`Unsupported optimiser type: ${settings.optimiserType}`);
    }

    return { x: xs, y: ys };
}

export function computeTrajectory2D(analysis, settings) {
    const numSteps = settings.numSteps ?? DEFAULT_NUM_STEPS;
    const xs = new Array(numSteps + 1).fill(0);
    const ys = new Array(numSteps + 1).fill(0);
    const zs = new Array(numSteps + 1).fill(0);

    xs[0] = settings.initialX;
    ys[0] = settings.initialY;
    zs[0] = ensureValue(evaluateNumber(analysis.fn, { x: xs[0], y: ys[0] }), 0);

    if (settings.optimiserType === "Gradient Descent") {
        for (let i = 0; i < numSteps; i += 1) {
            const gradX = evaluateNumber(analysis.gradient.x, { x: xs[i], y: ys[i] });
            const gradY = evaluateNumber(analysis.gradient.y, { x: xs[i], y: ys[i] });
            if (!Number.isFinite(gradX) || !Number.isFinite(gradY)) {
                xs[i + 1] = xs[i];
                ys[i + 1] = ys[i];
                zs[i + 1] = zs[i];
                continue;
            }

            const momentumX = i === 0 ? 0 : settings.momentum * (xs[i] - xs[i - 1]);
            const momentumY = i === 0 ? 0 : settings.momentum * (ys[i] - ys[i - 1]);

            xs[i + 1] = xs[i] - settings.learningRate * gradX + momentumX;
            ys[i + 1] = ys[i] - settings.learningRate * gradY + momentumY;
            zs[i + 1] = ensureValue(evaluateNumber(analysis.fn, { x: xs[i + 1], y: ys[i + 1] }), zs[i]);
        }
    } else if (settings.optimiserType === "Adam") {
        let mx = 0;
        let my = 0;
        let vx = 0;
        let vy = 0;

        for (let i = 0; i < numSteps; i += 1) {
            const gradX = evaluateNumber(analysis.gradient.x, { x: xs[i], y: ys[i] });
            const gradY = evaluateNumber(analysis.gradient.y, { x: xs[i], y: ys[i] });
            if (!Number.isFinite(gradX) || !Number.isFinite(gradY)) {
                xs[i + 1] = xs[i];
                ys[i + 1] = ys[i];
                zs[i + 1] = zs[i];
                continue;
            }

            mx = settings.beta1 * mx + (1 - settings.beta1) * gradX;
            my = settings.beta1 * my + (1 - settings.beta1) * gradY;
            vx = settings.beta2 * vx + (1 - settings.beta2) * gradX * gradX;
            vy = settings.beta2 * vy + (1 - settings.beta2) * gradY * gradY;

            const beta1Correction = 1 - settings.beta1 ** (i + 1);
            const beta2Correction = 1 - settings.beta2 ** (i + 1);
            const mxHat = mx / (Math.abs(beta1Correction) > EPSILON ? beta1Correction : EPSILON);
            const myHat = my / (Math.abs(beta1Correction) > EPSILON ? beta1Correction : EPSILON);
            const vxHat = vx / (Math.abs(beta2Correction) > EPSILON ? beta2Correction : EPSILON);
            const vyHat = vy / (Math.abs(beta2Correction) > EPSILON ? beta2Correction : EPSILON);

            xs[i + 1] = xs[i] - settings.learningRate * mxHat / (Math.sqrt(vxHat) + ADAM_EPSILON);
            ys[i + 1] = ys[i] - settings.learningRate * myHat / (Math.sqrt(vyHat) + ADAM_EPSILON);
            zs[i + 1] = ensureValue(evaluateNumber(analysis.fn, { x: xs[i + 1], y: ys[i + 1] }), zs[i]);
        }
    } else if (settings.optimiserType === "Newton") {
        for (let i = 0; i < numSteps; i += 1) {
            const gradX = evaluateNumber(analysis.gradient.x, { x: xs[i], y: ys[i] });
            const gradY = evaluateNumber(analysis.gradient.y, { x: xs[i], y: ys[i] });
            const hxx = evaluateNumber(analysis.hessian.xx, { x: xs[i], y: ys[i] });
            const hxy = evaluateNumber(analysis.hessian.xy, { x: xs[i], y: ys[i] });
            const hyx = evaluateNumber(analysis.hessian.yx, { x: xs[i], y: ys[i] });
            const hyy = evaluateNumber(analysis.hessian.yy, { x: xs[i], y: ys[i] });

            if ([gradX, gradY, hxx, hxy, hyx, hyy].some((v) => !Number.isFinite(v))) {
                xs[i + 1] = xs[i];
                ys[i + 1] = ys[i];
                zs[i + 1] = zs[i];
                continue;
            }

            const inv = invert2x2(hxx, hxy, hyx, hyy);
            if (!inv) {
                xs[i + 1] = xs[i];
                ys[i + 1] = ys[i];
                zs[i + 1] = zs[i];
                continue;
            }

            const stepX = inv.xx * gradX + inv.xy * gradY;
            const stepY = inv.yx * gradX + inv.yy * gradY;

            xs[i + 1] = xs[i] - stepX;
            ys[i + 1] = ys[i] - stepY;
            zs[i + 1] = ensureValue(evaluateNumber(analysis.fn, { x: xs[i + 1], y: ys[i + 1] }), zs[i]);
        }
    } else {
        throw new Error(`Unsupported optimiser type: ${settings.optimiserType}`);
    }

    return { x: xs, y: ys, z: zs };
}
