using System;
using System.Collections.Generic;

namespace NMicrograd
{
    /// <summary>
    /// A single weighted sum plus optional ReLU.
    /// </summary>
    public class Neuron : Module
    {
        private readonly List<Value> _weights;
        private readonly Value _bias;
        private readonly bool _nonlinear;

        public Neuron(int inputCount, bool nonlinear = true, Random? random = null)
        {
            if (inputCount < 1)
            {
                throw new ArgumentOutOfRangeException(nameof(inputCount), "A neuron needs at least one input.");
            }

            random = random ?? new Random();
            _weights = new List<Value>(inputCount);
            for (var index = 0; index < inputCount; index++)
            {
                var weight = random.NextDouble() * 2.0 - 1.0;
                _weights.Add(new Value(weight));
            }

            _bias = new Value(0.0);
            _nonlinear = nonlinear;
        }

        public Value Forward(IReadOnlyList<Value> inputs)
        {
            if (inputs == null)
            {
                throw new ArgumentNullException(nameof(inputs));
            }

            if (inputs.Count != _weights.Count)
            {
                throw new ArgumentException("Input count must match the neuron weight count.", nameof(inputs));
            }

            var activation = _bias;
            for (var index = 0; index < _weights.Count; index++)
            {
                activation = activation + _weights[index] * inputs[index];
            }

            return _nonlinear ? activation.Relu() : activation;
        }

        public override IReadOnlyList<Value> Parameters()
        {
            var parameters = new List<Value>(_weights.Count + 1);
            parameters.AddRange(_weights);
            parameters.Add(_bias);
            return parameters;
        }

        public override string ToString()
        {
            var kind = _nonlinear ? "ReLU" : "Linear";
            return $"{kind}Neuron({_weights.Count})";
        }
    }
}
