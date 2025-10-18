import { Univariate } from "./univariate.js";
import { Bivariate } from "./bivariate.js";

function setupTabs() {
    const buttons = Array.from(document.querySelectorAll(".tab-button"));
    const contents = Array.from(document.querySelectorAll(".tab-content"));

    buttons.forEach((button) => {
        button.addEventListener("click", () => {
            const targetId = button.dataset.target;
            buttons.forEach((btn) => btn.classList.toggle("active", btn === button));
            contents.forEach((section) => {
                section.classList.toggle("active", section.id === targetId);
            });
        });
    });
}

function disableScrollOnNumberInputs() {
    const inputs = document.querySelectorAll('input[type="number"]');

    inputs.forEach(input => {
        input.addEventListener('wheel', e => e.preventDefault());

        input.addEventListener('keydown', e => {
            if (e.key === 'ArrowUp' || e.key === 'ArrowDown') e.preventDefault();
        });
    });
}

function selectElements() {
    const univariate = new Univariate({
        plot: document.getElementById("univariate-plot"),
        functionInput: document.getElementById("univariate-function"),
        errorEl: document.getElementById("univariate-error"),
        gradientOutput: document.getElementById("univariate-derivative"),
        hessianField: document.querySelector("#univariate-tab [data-role='hessian']"),
        hessianOutput: document.getElementById("univariate-hessian"),
        optimiserInputs: Array.from(document.querySelectorAll("input[name='univariate-optimiser']")),
        initialXInput: document.getElementById("univariate-initial-x"),
        learningRateField: document.querySelector("#univariate-tab [data-role='learning-rate']"),
        learningRateInput: document.getElementById("univariate-learning-rate"),
        momentumField: document.querySelector("#univariate-tab [data-role='momentum']"),
        momentumInput: document.getElementById("univariate-momentum"),
        slider: document.getElementById("univariate-step"),
        sliderValue: document.getElementById("univariate-step-value"),
        stepButton: document.getElementById("univariate-step-button"),
    });

    const bivariate = new Bivariate({
        plot: document.getElementById("bivariate-plot"),
        functionInput: document.getElementById("bivariate-function"),
        errorEl: document.getElementById("bivariate-error"),
        gradientOutput: document.getElementById("bivariate-gradient"),
        hessianField: document.querySelector("#bivariate-tab [data-role='hessian']"),
        hessianOutput: document.getElementById("bivariate-hessian"),
        optimiserInputs: Array.from(document.querySelectorAll("input[name='bivariate-optimiser']")),
        initialXInput: document.getElementById("bivariate-initial-x"),
        initialYInput: document.getElementById("bivariate-initial-y"),
        learningRateField: document.querySelector("#bivariate-tab [data-role='learning-rate']"),
        learningRateInput: document.getElementById("bivariate-learning-rate"),
        momentumField: document.querySelector("#bivariate-tab [data-role='momentum']"),
        momentumInput: document.getElementById("bivariate-momentum"),
        beta1Field: document.querySelector("#bivariate-tab [data-role='beta1']"),
        beta1Input: document.getElementById("bivariate-beta1"),
        beta2Field: document.querySelector("#bivariate-tab [data-role='beta2']"),
        beta2Input: document.getElementById("bivariate-beta2"),
        slider: document.getElementById("bivariate-step"),
        sliderValue: document.getElementById("bivariate-step-value"),
        stepButton: document.getElementById("bivariate-step-button"),
    });

    return { univariate, bivariate };
}

document.addEventListener("DOMContentLoaded", () => {
    setupTabs();
    disableScrollOnNumberInputs();
    const { univariate, bivariate } = selectElements();
    univariate.init();
    bivariate.init();
});
