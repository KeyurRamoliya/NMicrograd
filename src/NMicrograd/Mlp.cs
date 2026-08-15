using System;
using System.Collections.Generic;
using System.Text;

namespace NMicrograd
{
    /// <summary>
    /// A stack of layers. Hidden layers use ReLU; the last layer is linear.
    /// </summary>
    public class Mlp : Module
    {
        private readonly List<Layer> _layers;

        public Mlp(int inputCount, IReadOnlyList<int> layerSizes, Random? random = null)
        {
            if (layerSizes == null)
            {
                throw new ArgumentNullException(nameof(layerSizes));
            }

            if (layerSizes.Count < 1)
            {
                throw new ArgumentException("An MLP needs at least one layer size.", nameof(layerSizes));
            }

            random = random ?? new Random();
            _layers = new List<Layer>(layerSizes.Count);

            var previousSize = inputCount;
            for (var index = 0; index < layerSizes.Count; index++)
            {
                var nonlinear = index != layerSizes.Count - 1;
                _layers.Add(new Layer(previousSize, layerSizes[index], nonlinear, random));
                previousSize = layerSizes[index];
            }
        }

        public IReadOnlyList<Value> Forward(IReadOnlyList<Value> inputs)
        {
            IReadOnlyList<Value> current = inputs;
            foreach (var layer in _layers)
            {
                current = layer.Forward(current);
            }

            return current;
        }

        public override IReadOnlyList<Value> Parameters()
        {
            var parameters = new List<Value>();
            foreach (var layer in _layers)
            {
                parameters.AddRange(layer.Parameters());
            }

            return parameters;
        }

        public override string ToString()
        {
            var builder = new StringBuilder();
            builder.Append("MLP of [");
            for (var index = 0; index < _layers.Count; index++)
            {
                if (index > 0)
                {
                    builder.Append(", ");
                }

                builder.Append(_layers[index]);
            }

            builder.Append(']');
            return builder.ToString();
        }
    }
}
