# NMicrograd

A tiny scalar automatic-differentiation engine and a small neural-network library on top of it. The engine builds a graph of individual adds, multiplies, powers, and ReLUs; `Backward` walks that graph in reverse and fills in gradients.

The library targets **.NET Standard 2.0**. Tests and the demo run on **.NET 10**.

## Build and test

```bash
dotnet test NMicrograd.slnx
```

## Train a small classifier

```bash
dotnet run --project samples/NMicrograd.Demo
```

This trains a 2 → 16 → 16 → 1 network on two interleaving half-circles (labels ±1) with a hinge loss, L2 regularization, and SGD. It prints loss and accuracy each step and writes `moons.svg` in the current working directory.

## Usage

```csharp
using NMicrograd;

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
Console.WriteLine($"{g.Data:F4}");
g.Backward();
Console.WriteLine($"{a.Grad:F4}");
Console.WriteLine($"{b.Grad:F4}");
```

```csharp
var model = new Mlp(2, new[] { 16, 16, 1 }, new Random(1337));
var score = model.Forward(new[] { new Value(1.0), new Value(-2.0) })[0];
model.ZeroGrad();
score.Backward();
foreach (var parameter in model.Parameters())
{
    parameter.Data -= 0.1 * parameter.Grad;
}
```

## Note

This repository is a .NET port of [micrograd](https://github.com/karpathy/micrograd) by Andrej Karpathy. Full credit for the design, the algorithm, and the original implementation belongs to that project.

## License

MIT
