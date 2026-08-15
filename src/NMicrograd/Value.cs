using System;
using System.Collections.Generic;

namespace NMicrograd
{
    /// <summary>
    /// A scalar that records how it was produced so gradients can be computed in reverse.
    /// </summary>
    public class Value
    {
        private readonly HashSet<Value> _parents;
        private Action _backward;

        public Value(double data)
            : this(data, Array.Empty<Value>())
        {
        }

        private Value(double data, IEnumerable<Value> parents)
        {
            Data = data;
            Grad = 0.0;
            _parents = new HashSet<Value>(parents);
            _backward = () => { };
        }

        public double Data { get; set; }

        public double Grad { get; set; }

        public static implicit operator Value(double data)
        {
            return new Value(data);
        }

        public static Value operator +(Value left, Value right)
        {
            var result = new Value(left.Data + right.Data, [left, right]);
            result._backward = () =>
            {
                left.Grad += result.Grad;
                right.Grad += result.Grad;
            };
            return result;
        }

        public static Value operator *(Value left, Value right)
        {
            var result = new Value(left.Data * right.Data, [left, right]);
            result._backward = () =>
            {
                left.Grad += right.Data * result.Grad;
                right.Grad += left.Data * result.Grad;
            };
            return result;
        }

        public static Value operator -(Value value)
        {
            return value * -1.0;
        }

        public static Value operator -(Value left, Value right)
        {
            return left + (-right);
        }

        public static Value operator /(Value left, Value right)
        {
            return left * right.Pow(-1.0);
        }

        public Value Pow(double exponent)
        {
            var result = new Value(Math.Pow(Data, exponent), [this]);
            result._backward = () =>
            {
                Grad += exponent * Math.Pow(Data, exponent - 1.0) * result.Grad;
            };
            return result;
        }

        public Value Relu()
        {
            var result = new Value(Data < 0.0 ? 0.0 : Data, [this]);
            result._backward = () =>
            {
                Grad += (result.Data > 0.0 ? 1.0 : 0.0) * result.Grad;
            };
            return result;
        }

        public void Backward()
        {
            var order = new List<Value>();
            var visited = new HashSet<Value>();
            BuildTopology(this, visited, order);

            Grad = 1.0;
            for (var index = order.Count - 1; index >= 0; index--)
            {
                order[index]._backward();
            }
        }

        public override string ToString()
        {
            return $"Value(data={Data}, grad={Grad})";
        }

        private static void BuildTopology(Value node, HashSet<Value> visited, List<Value> order)
        {
            if (!visited.Add(node))
            {
                return;
            }

            foreach (var parent in node._parents)
            {
                BuildTopology(parent, visited, order);
            }

            order.Add(node);
        }
    }
}
