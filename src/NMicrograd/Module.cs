using System;
using System.Collections.Generic;

namespace NMicrograd
{
    /// <summary>
    /// Base type for network pieces that expose parameters and can clear their gradients.
    /// </summary>
    public class Module
    {
        public void ZeroGrad()
        {
            foreach (var parameter in Parameters())
            {
                parameter.Grad = 0.0;
            }
        }

        public virtual IReadOnlyList<Value> Parameters()
        {
            return Array.Empty<Value>();
        }
    }
}
