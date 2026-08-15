using System;
using System.Collections.Generic;
using System.Text;

namespace NMicrograd
{
    /// <summary>
    /// A row of neurons that all see the same inputs.
    /// </summary>
    public class Layer : Module
    {
        private readonly List<Neuron> _neurons;

        public Layer(int inputCount, int outputCount, bool nonlinear = true, Random? random = null)
        {
            if (outputCount < 1)
            {
                throw new ArgumentOutOfRangeException(nameof(outputCount), "A layer needs at least one neuron.");
            }

            random = random ?? new Random();
            _neurons = new List<Neuron>(outputCount);
            for (var index = 0; index < outputCount; index++)
            {
                _neurons.Add(new Neuron(inputCount, nonlinear, random));
            }
        }

        public IReadOnlyList<Value> Forward(IReadOnlyList<Value> inputs)
        {
            var outputs = new List<Value>(_neurons.Count);
            foreach (var neuron in _neurons)
            {
                outputs.Add(neuron.Forward(inputs));
            }

            return outputs;
        }

        public override IReadOnlyList<Value> Parameters()
        {
            var parameters = new List<Value>();
            foreach (var neuron in _neurons)
            {
                parameters.AddRange(neuron.Parameters());
            }

            return parameters;
        }

        public override string ToString()
        {
            var builder = new StringBuilder();
            builder.Append("Layer of [");
            for (var index = 0; index < _neurons.Count; index++)
            {
                if (index > 0)
                {
                    builder.Append(", ");
                }

                builder.Append(_neurons[index]);
            }

            builder.Append(']');
            return builder.ToString();
        }
    }
}
