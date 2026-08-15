namespace NMicrograd.Tests
{
    public class EngineTests
    {
        private const double Tolerance = 1e-6;

        [Fact]
        public void SanityCheck_MatchesExpectedForwardAndGradient()
        {
            var x = new Value(-4.0);
            var z = 2 * x + 2 + x;
            var q = z.Relu() + z * x;
            var h = (z * z).Relu();
            var y = h + q + q * x;
            y.Backward();

            AssertClose(-20.0, y.Data);
            AssertClose(46.0, x.Grad);
        }

        [Fact]
        public void MoreOps_MatchesExpectedForwardAndGradients()
        {
            var a = new Value(-4.0);
            var b = new Value(2.0);
            var c = a + b;
            var d = a * b + b.Pow(3);
            c += c + 1;
            c += 1 + c + (-a);
            d += d * 2 + (b + a).Relu();
            d += 3 * d + (b - a).Relu();
            var e = c - d;
            var f = e.Pow(2);
            var g = f / 2.0;
            g += 10.0 / f;
            g.Backward();

            AssertClose(24.70408163265306, g.Data);
            AssertClose(138.83381924198252, a.Grad);
            AssertClose(645.5772594752186, b.Grad);
        }

        [Fact]
        public void ImplicitDouble_AddsOnEitherSide()
        {
            var a = new Value(3.0);

            var left = 1.0 + a;
            var right = a + 1.0;

            AssertClose(4.0, left.Data);
            AssertClose(4.0, right.Data);
        }

        [Fact]
        public void RepeatedUse_AccumulatesGradient()
        {
            var a = new Value(2.0);
            var sum = a + a;
            sum.Backward();

            AssertClose(2.0, a.Grad);
        }

        [Fact]
        public void Relu_BlocksGradientWhenNegative()
        {
            var a = new Value(-3.0);
            var activated = a.Relu();
            activated.Backward();

            AssertClose(0.0, activated.Data);
            AssertClose(0.0, a.Grad);
        }

        private static void AssertClose(double expected, double actual)
        {
            Assert.True(
                Math.Abs(expected - actual) < Tolerance,
                $"Expected {expected}, got {actual}.");
        }
    }
}
