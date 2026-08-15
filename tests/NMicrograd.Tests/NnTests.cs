namespace NMicrograd.Tests
{
    public class NnTests
    {
        [Fact]
        public void Mlp_HasExpectedParameterCount()
        {
            var model = new Mlp(2, new[] { 4, 1 }, new Random(0));

            Assert.Equal(17, model.Parameters().Count);
        }

        [Fact]
        public void Mlp_BackwardFillsParameterGradients()
        {
            var model = new Mlp(2, new[] { 4, 1 }, new Random(0));
            var inputs = new[] { new Value(1.0), new Value(-2.0) };

            var score = model.Forward(inputs)[0];
            score.Backward();

            Assert.Contains(model.Parameters(), parameter => Math.Abs(parameter.Grad) > 0.0);
        }

        [Fact]
        public void Mlp_ZeroGradClearsParameterGradients()
        {
            var model = new Mlp(2, new[] { 4, 1 }, new Random(0));
            var inputs = new[] { new Value(1.0), new Value(-2.0) };

            var score = model.Forward(inputs)[0];
            score.Backward();
            model.ZeroGrad();

            Assert.All(model.Parameters(), parameter => Assert.Equal(0.0, parameter.Grad));
        }
    }
}
