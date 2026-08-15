namespace NMicrograd.Demo
{
    internal sealed class MoonSample(double x, double y, double label)
    {
        public double X { get; } = x;

        public double Y { get; } = y;

        public double Label { get; } = label;
    }

    internal static class Moons
    {
        public static IReadOnlyList<MoonSample> Generate(int sampleCount, double noise, Random random)
        {
            var outerCount = sampleCount / 2;
            var innerCount = sampleCount - outerCount;
            var samples = new List<MoonSample>(sampleCount);

            AddArc(samples, outerCount, 1.0, 0.0, 1.0, 0.0, 1.0, noise, random);
            AddArc(samples, innerCount, -1.0, 1.0, -1.0, 0.5, -1.0, noise, random);

            return samples;
        }

        private static void AddArc(
            List<MoonSample> samples,
            int count,
            double label,
            double offsetX,
            double scaleX,
            double offsetY,
            double scaleY,
            double noise,
            Random random)
        {
            for (var index = 0; index < count; index++)
            {
                var angle = Math.PI * index / Math.Max(count - 1, 1);
                var x = offsetX + scaleX * Math.Cos(angle) + NextNoise(random, noise);
                var y = offsetY + scaleY * Math.Sin(angle) + NextNoise(random, noise);
                samples.Add(new MoonSample(x, y, label));
            }
        }

        private static double NextNoise(Random random, double noise)
        {
            var u1 = 1.0 - random.NextDouble();
            var u2 = 1.0 - random.NextDouble();
            var gaussian = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
            return gaussian * noise;
        }
    }
}
