import { analyzeFunction2D, computeTrajectory2D, DEFAULT_NUM_STEPS } from "./optimisers.js";

const Plotly = window.Plotly;

if (!Plotly) {
    throw new Error("Plotly is required for plotting.");
}

function debounce(fn, wait) {
    let timeoutId = null;
    return function debounced(...args) {
        window.clearTimeout(timeoutId);
        timeoutId = window.setTimeout(() => fn.apply(this, args), wait);
    };
}

export class Bivariate {
    constructor(elements) {
        this.plotEl = elements.plot;
        this.functionInput = elements.functionInput;
        this.errorEl = elements.errorEl;
        this.gradientOutput = elements.gradientOutput;
        this.hessianField = elements.hessianField;
        this.hessianOutput = elements.hessianOutput;
        this.optimiserInputs = elements.optimiserInputs;
        this.initialXInput = elements.initialXInput;
        this.initialYInput = elements.initialYInput;
        this.learningRateField = elements.learningRateField;
        this.learningRateInput = elements.learningRateInput;
        this.momentumField = elements.momentumField;
        this.momentumInput = elements.momentumInput;
        this.beta1Field = elements.beta1Field;
        this.beta1Input = elements.beta1Input;
        this.beta2Field = elements.beta2Field;
        this.beta2Input = elements.beta2Input;
        this.slider = elements.slider;
        this.sliderValue = elements.sliderValue;
        this.stepButton = elements.stepButton;

        this.optimiserType = "Gradient Descent";
        this.initialX = 0.5;
        this.initialY = 0.5;
        this.learningRate = 0.1;
        this.momentum = 0;
        this.beta1 = 0.9;
        this.beta2 = 0.999;
        this.numSteps = DEFAULT_NUM_STEPS;
        this.currentStep = 0;
        this.analysis = null;
        this.trajectory = null;
    }

    init() {
        this.bindEvents();
        this.refreshFromInputs();
    }

    bindEvents() {
        this.functionInput.addEventListener("input", debounce(() => {
            this.handleFunctionChange();
        }, 300));

        this.optimiserInputs.forEach((input) => {
            input.addEventListener("change", () => {
                if (input.checked) {
                    this.optimiserType = input.value;
                    this.updateParameterVisibility();
                    this.recomputeTrajectory();
                }
            });
        });

        this.initialXInput.addEventListener("change", () => {
            this.initialX = Number(this.initialXInput.value);
            this.recomputeTrajectory();
        });
        this.initialYInput.addEventListener("change", () => {
            this.initialY = Number(this.initialYInput.value);
            this.recomputeTrajectory();
        });

        this.learningRateInput.addEventListener("change", () => {
            this.learningRate = Number(this.learningRateInput.value);
            this.recomputeTrajectory();
        });

        this.momentumInput.addEventListener("change", () => {
            this.momentum = Number(this.momentumInput.value);
            this.recomputeTrajectory();
        });

        this.beta1Input.addEventListener("change", () => {
            this.beta1 = Number(this.beta1Input.value);
            this.recomputeTrajectory();
        });

        this.beta2Input.addEventListener("change", () => {
            this.beta2 = Number(this.beta2Input.value);
            this.recomputeTrajectory();
        });

        this.slider.addEventListener("input", () => {
            this.currentStep = Number(this.slider.value);
            this.updateStepDisplay();
            this.renderPlot();
        });

        this.stepButton.addEventListener("click", () => {
            if (this.currentStep < this.numSteps) {
                this.currentStep += 1;
                this.slider.value = String(this.currentStep);
                this.updateStepDisplay();
                this.renderPlot();
            }
            this.updateStepButton();
        });
    }

    refreshFromInputs() {
        this.initialX = Number(this.initialXInput.value);
        this.initialY = Number(this.initialYInput.value);
        this.learningRate = Number(this.learningRateInput.value);
        this.momentum = Number(this.momentumInput.value);
        this.beta1 = Number(this.beta1Input.value);
        this.beta2 = Number(this.beta2Input.value);
        this.handleFunctionChange();
    }

    handleFunctionChange() {
        const functionStr = this.functionInput.value.trim();
        if (functionStr.length === 0) {
            this.setError("Function cannot be empty.");
            return;
        }

        try {
            this.analysis = analyzeFunction2D(functionStr);
            this.clearError();
            this.gradientOutput.value = this.formatGradient();
            this.hessianOutput.value = this.formatHessian();
            this.updateParameterVisibility();
            this.recomputeTrajectory();
        } catch (error) {
            this.setError(error.message);
        }
    }

    formatGradient() {
        if (!this.analysis) {
            return "";
        }
        const { x, y } = this.analysis.gradientNodes;
        return `[${x.toString()}, ${y.toString()}]`;
    }

    formatHessian() {
        if (!this.analysis) {
            return "";
        }
        const { xx, xy, yx, yy } = this.analysis.hessianNodes;
        return `[[${xx.toString()}, ${xy.toString()}], [${yx.toString()}, ${yy.toString()}]]`;
    }

    updateParameterVisibility() {
        const isGD = this.optimiserType === "Gradient Descent";
        const isAdam = this.optimiserType === "Adam";
        const isNewton = this.optimiserType === "Newton";

        this.learningRateField.classList.toggle("hidden", isNewton);
        this.momentumField.classList.toggle("hidden", !isGD);
        this.beta1Field.classList.toggle("hidden", !isAdam);
        this.beta2Field.classList.toggle("hidden", !isAdam);
        this.hessianField.classList.toggle("hidden", !isNewton);
    }

    recomputeTrajectory() {
        if (!this.analysis) {
            return;
        }

        try {
            this.trajectory = computeTrajectory2D(this.analysis, {
                optimiserType: this.optimiserType,
                initialX: this.initialX,
                initialY: this.initialY,
                learningRate: this.learningRate,
                momentum: this.momentum,
                beta1: this.beta1,
                beta2: this.beta2,
                numSteps: this.numSteps,
            });
            this.currentStep = 0;
            this.slider.max = String(this.numSteps);
            this.slider.value = "0";
            this.updateStepDisplay();
            this.updateStepButton();
            this.renderPlot();
        } catch (error) {
            this.setError(error.message);
        }
    }

    renderPlot() {
        if (!this.analysis || !this.trajectory) {
            return;
        }

        const stepIdx = Math.min(this.currentStep, this.trajectory.x.length - 1);
        const contour = this.generateContourData();
        const data = [
            {
                x: contour.x,
                y: contour.y,
                z: contour.z,
                type: "contour",
                colorscale: "Viridis",
                contours: { coloring: "heatmap" },
                hoverinfo: "skip",
                showscale: true,
                ncontours: 40,
                name: "Surface",
            },
            {
                x: this.trajectory.x.slice(0, stepIdx + 1),
                y: this.trajectory.y.slice(0, stepIdx + 1),
                type: "scatter",
                mode: "lines+markers",
                line: { color: "#d65c5c", width: 2 },
                marker: { color: "#d65c5c", size: 6 },
                name: "Trajectory",
            },
            {
                x: [this.trajectory.x[stepIdx]],
                y: [this.trajectory.y[stepIdx]],
                type: "scatter",
                mode: "markers",
                marker: { color: "#c40018", size: 10 },
                name: "Current step",
            },
        ];

        const layout = {
            margin: { l: 55, r: 20, t: 10, b: 45 },
            xaxis: { title: "x" },
            yaxis: { title: "y" },
            showlegend: false,
            hovermode: false,
        };

        const config = { responsive: true, displayModeBar: false };
        Plotly.react(this.plotEl, data, layout, config);
    }

    generateContourData() {
        const gridSize = 60;
        const trajX = this.trajectory 
            ? this.trajectory.x.slice(0, this.currentStep + 1) 
            : [ -1, 1 ];
        const trajY = this.trajectory 
            ? this.trajectory.y.slice(0, this.currentStep + 1) 
            : [ -1, 1 ];
        const maxAbsX = trajX.reduce((acc, value) => Math.max(acc, Math.abs(value)), 1);
        const maxAbsY = trajY.reduce((acc, value) => Math.max(acc, Math.abs(value)), 1);
        const radius = Math.max(1, maxAbsX, maxAbsY);
        const limit = radius > 1 ? 1.2 * radius : 1;
        const minVal = -limit;
        const maxVal = limit;
        console.log(limit)

        const xs = [];
        const ys = [];
        for (let i = 0; i < gridSize; i += 1) {
            const frac = i / (gridSize - 1);
            xs.push(minVal + frac * (maxVal - minVal));
            ys.push(minVal + frac * (maxVal - minVal));
        }

        const z = ys.map((yVal) => {
            const row = [];
            for (let i = 0; i < xs.length; i += 1) {
                row.push(this.evaluateFunction({ x: xs[i], y: yVal }));
            }
            return row;
        });

        return { x: xs, y: ys, z };
    }

    evaluateFunction(scope) {
        if (!this.analysis) {
            return null;
        }
        try {
            const value = this.analysis.fn.evaluate(scope);
            const numeric = typeof value === "number" ? value : Number(value);
            return Number.isFinite(numeric) ? numeric : null;
        } catch (error) {
            return null;
        }
    }

    updateStepDisplay() {
        this.sliderValue.textContent = String(this.currentStep);
        this.updateStepButton();
    }

    updateStepButton() {
        const disabled = this.currentStep >= this.numSteps;
        this.stepButton.disabled = disabled;
    }

    setError(message) {
        this.errorEl.textContent = message;
    }

    clearError() {
        this.errorEl.textContent = "";
    }
}
