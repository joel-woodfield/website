import { analyzeFunction1D, computeTrajectory1D, DEFAULT_NUM_STEPS } from "./optimisers.js";

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

export class Univariate {
    constructor(elements) {
        this.plotEl = elements.plot;
        this.functionInput = elements.functionInput;
        this.errorEl = elements.errorEl;
        this.gradientOutput = elements.gradientOutput;
        this.hessianField = elements.hessianField;
        this.hessianOutput = elements.hessianOutput;
        this.optimiserInputs = elements.optimiserInputs;
        this.initialXInput = elements.initialXInput;
        this.learningRateField = elements.learningRateField;
        this.learningRateInput = elements.learningRateInput;
        this.momentumField = elements.momentumField;
        this.momentumInput = elements.momentumInput;
        this.slider = elements.slider;
        this.sliderValue = elements.sliderValue;
        this.stepButton = elements.stepButton;

        this.optimiserType = "Gradient Descent";
        this.initialX = 0.5;
        this.learningRate = 0.1;
        this.momentum = 0;
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

        this.learningRateInput.addEventListener("change", () => {
            this.learningRate = Number(this.learningRateInput.value);
            this.recomputeTrajectory();
        });

        this.momentumInput.addEventListener("change", () => {
            this.momentum = Number(this.momentumInput.value);
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
        this.learningRate = Number(this.learningRateInput.value);
        this.momentum = Number(this.momentumInput.value);
        this.handleFunctionChange();
    }

    handleFunctionChange() {
        const functionStr = this.functionInput.value.trim();
        if (functionStr.length === 0) {
            this.setError("Function cannot be empty.");
            return;
        }

        try {
            this.analysis = analyzeFunction1D(functionStr);
            this.clearError();
            this.gradientOutput.value = this.analysis.gradientNode.toString();
            this.hessianOutput.value = this.analysis.hessianNode.toString();
            this.updateParameterVisibility();
            this.recomputeTrajectory();
        } catch (error) {
            this.setError(error.message);
        }
    }

    updateParameterVisibility() {
        const isGradientDescent = this.optimiserType === "Gradient Descent";
        this.learningRateField.classList.toggle("hidden", !isGradientDescent);
        this.momentumField.classList.toggle("hidden", !isGradientDescent);
        this.hessianField.classList.toggle("hidden", isGradientDescent);
    }

    recomputeTrajectory() {
        if (!this.analysis) {
            return;
        }

        try {
            this.trajectory = computeTrajectory1D(this.analysis, {
                optimiserType: this.optimiserType,
                initialX: this.initialX,
                learningRate: this.learningRate,
                momentum: this.momentum,
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
        const samples = this.generateFunctionSamples();

        const data = [
            {
                x: samples.x,
                y: samples.y,
                type: "scatter",
                mode: "lines",
                line: { color: "#1f77b4", width: 2 },
                hoverinfo: "skip",
                name: "Function",
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
            yaxis: { title: "f(x)" },
            showlegend: false,
            hovermode: false,
        };

        const config = { responsive: true, displayModeBar: false };
        Plotly.react(this.plotEl, data, layout, config);
    }

    generateFunctionSamples() {
        const sampleCount = 200;
        const trajectoryX = this.trajectory 
            ? this.trajectory.x.slice(0, this.currentStep + 1) 
            : [ -1, 1 ];
        const maxAbsX = trajectoryX.reduce((acc, value) => Math.max(acc, Math.abs(value)), 1);
        const radius = Math.max(1, maxAbsX);
        const limit = radius > 1 ? 1.2 * radius : 1;
        const xMin = -limit;
        const xMax = limit;

        const xs = [];
        const ys = [];
        for (let i = 0; i < sampleCount; i += 1) {
            const xVal = xMin + (i / (sampleCount - 1)) * (xMax - xMin);
            xs.push(xVal);
            ys.push(this.evaluateFunction({ x: xVal }));
        }

        return { x: xs, y: ys };
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
