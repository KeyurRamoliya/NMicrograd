using System.Globalization;
using System.Text;

namespace NMicrograd.Demo
{
    internal static class Program
    {
        private const int SampleCount = 100;
        private const int StepCount = 100;
        private const double Regularization = 1e-4;
        private const string SvgPath = "moons.svg";

        public static void Main()
        {
            var random = new Random(1337);
            var samples = Moons.Generate(SampleCount, 0.1, random);
            var model = new Mlp(2, [16, 16, 1], random);

            Console.WriteLine(model);
            Console.WriteLine($"number of parameters {model.Parameters().Count}");

            for (var step = 0; step < StepCount; step++)
            {
                var (loss, accuracy) = Evaluate(model, samples);
                model.ZeroGrad();
                loss.Backward();

                var learningRate = 1.0 - 0.9 * step / StepCount;
                foreach (var parameter in model.Parameters())
                {
                    parameter.Data -= learningRate * parameter.Grad;
                }

                Console.WriteLine($"step {step} loss {loss.Data}, accuracy {accuracy * 100.0}%");
            }

            File.WriteAllText(SvgPath, RenderDecisionBoundary(model, samples));
            Console.WriteLine($"wrote {Path.GetFullPath(SvgPath)}");
        }

        private static (Value Loss, double Accuracy) Evaluate(Mlp model, IReadOnlyList<MoonSample> samples)
        {
            var losses = new List<Value>(samples.Count);
            var correct = 0;

            foreach (var sample in samples)
            {
                var inputs = new[] { new Value(sample.X), new Value(sample.Y) };
                var score = model.Forward(inputs)[0];
                var margin = 1.0 + -sample.Label * score;
                losses.Add(margin.Relu());

                var predictedPositive = score.Data > 0.0;
                var labeledPositive = sample.Label > 0.0;
                if (predictedPositive == labeledPositive)
                {
                    correct++;
                }
            }

            var dataLoss = Mean(losses);
            var regLoss = Regularization * SumSquares(model.Parameters());
            var accuracy = (double)correct / samples.Count;
            return (dataLoss + regLoss, accuracy);
        }

        private static Value Mean(IReadOnlyList<Value> values)
        {
            var total = new Value(0.0);
            foreach (var value in values)
            {
                total = total + value;
            }

            return total * (1.0 / values.Count);
        }

        private static Value SumSquares(IReadOnlyList<Value> parameters)
        {
            var total = new Value(0.0);
            foreach (var parameter in parameters)
            {
                total = total + parameter * parameter;
            }

            return total;
        }

        private static string RenderDecisionBoundary(Mlp model, IReadOnlyList<MoonSample> samples)
        {
            const int pixels = 480;
            const int grid = 80;
            const double padding = 1.0;

            var minX = samples.Min(sample => sample.X) - padding;
            var maxX = samples.Max(sample => sample.X) + padding;
            var minY = samples.Min(sample => sample.Y) - padding;
            var maxY = samples.Max(sample => sample.Y) + padding;

            var builder = new StringBuilder();
            builder.AppendLine($"<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{pixels}\" height=\"{pixels}\" viewBox=\"0 0 {pixels} {pixels}\">");
            builder.AppendLine("<rect width=\"100%\" height=\"100%\" fill=\"#111827\"/>");

            var cell = (double)pixels / grid;
            for (var row = 0; row < grid; row++)
            {
                for (var column = 0; column < grid; column++)
                {
                    var x = minX + (column + 0.5) / grid * (maxX - minX);
                    var y = maxY - (row + 0.5) / grid * (maxY - minY);
                    var score = model.Forward([new Value(x), new Value(y)])[0];
                    var fill = score.Data > 0.0 ? "#7f1d1d" : "#1e3a5f";
                    builder.AppendLine(
                        $"<rect x=\"{Format(column * cell)}\" y=\"{Format(row * cell)}\" width=\"{Format(cell)}\" height=\"{Format(cell)}\" fill=\"{fill}\"/>");
                }
            }

            foreach (var sample in samples)
            {
                var px = (sample.X - minX) / (maxX - minX) * pixels;
                var py = (maxY - sample.Y) / (maxY - minY) * pixels;
                var fill = sample.Label > 0.0 ? "#f97316" : "#38bdf8";
                builder.AppendLine(
                    $"<circle cx=\"{Format(px)}\" cy=\"{Format(py)}\" r=\"4\" fill=\"{fill}\" stroke=\"#0f172a\" stroke-width=\"1\"/>");
            }

            builder.AppendLine("</svg>");
            return builder.ToString();
        }

        private static string Format(double value)
        {
            return value.ToString("0.###", CultureInfo.InvariantCulture);
        }
    }
}
