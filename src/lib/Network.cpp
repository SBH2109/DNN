#include <Network.h>

#include <iostream>
#include <stdexcept>

namespace dnn {

Network::~Network() {
    for (Layer* layer : layers) {
        delete layer;
    }
    delete ownedLossFunction;
}

Network::Network(std::vector<std::tuple<std::vector<std::vector<double> >, std::vector<double>, ActivationFunction *> > layerParameters, LossFunction *lossFunction) : ownedLossFunction(lossFunction) {
    layers.reserve(layerParameters.size());
    for (uint64_t i = 0; i < layerParameters.size(); ++i) {
        auto& [weights, bias, activationFunction] = layerParameters[i];
        const uint64_t outputDim = weights.size();
        const uint64_t inputDim = outputDim > 0 ? weights[0].size() : 0;

        Layer* layer = new Layer(i, nextSeed++, inputDim, outputDim, activationFunction);
        layer->setParameters(std::move(weights), std::move(bias));
        layers.push_back(layer);
    }
}

Network &Network::addLayer(uint64_t inputDim, uint64_t outputDim, ActivationFunction *activationFunction) {
    const uint64_t id = layers.size();
    Layer* layer = new Layer(id, nextSeed++, inputDim, outputDim, activationFunction);
    layer->init();
    layers.push_back(layer);
    return *this;
}

const std::vector<double> &Network::getLayerInput(uint64_t layerId) const {
    if (layerId >= layers.size()) {
        throw std::out_of_range("Invalid layer id");
    }
    if (layerId == 0) {
        return initialInput;
    }
    return layers[layerId - 1]->getActivationOutput();
}

void Network::fullForwardPropagation(const std::vector<double>& input) {
    initialInput = input;
    if (layers.empty()) {
        output = initialInput;
        return;
    }
    layers[0]->forwardPropagation(initialInput);
    for (std::size_t i = 1; i < layers.size(); ++i) {
        layers[i]->forwardPropagation(layers[i - 1]->getActivationOutput());
    }
    output = layers.back()->getActivationOutput();
}

void Network::fullBackwardPropagation(const std::vector<double> &predictedOutput, const std::vector<double> &actualOutput) {
    if (ownedLossFunction == nullptr) {
        throw std::logic_error("Invalid loss function");
    }
    if (layers.empty()) {
        return;
    }
    std::vector<double> dACurr = ownedLossFunction->getGradient(predictedOutput, actualOutput);
    for (std::size_t i = layers.size(); i > 0; --i) {
        const std::size_t j = i - 1;
        layers[j]->backwardPropagation(dACurr, getLayerInput(j));
        dACurr = layers[j]->getDPreviousLayerOutput();
    }
}


void Network::setLossFunction(LossFunction* lossFunction) {
    delete this->ownedLossFunction;
    this->ownedLossFunction = lossFunction;
}

const LossFunction* Network::getLossFunction() const {
    return ownedLossFunction;
}

const std::vector<double>& Network::getOutput() const {
    return output;
}

const std::vector<double>& Network::getCostHistory() const {
    return costHistory;
}

const std::vector<double>& Network::getAccuracyHistory() const {
    return accuracyHistory;
}

const std::vector<Layer*>& Network::getLayers() const {
    return layers;
}

void Network::setLayers(std::vector<Layer*> newLayers) {
    for (Layer* layer : layers) {
        delete layer;
    }
    layers = std::move(newLayers);
    initialInput.clear();
    output.clear();
    costHistory.clear();
    accuracyHistory.clear();
}

}
