using DreamHopper.UI.Views;
using Rhino;
using System;
using System.Runtime.InteropServices;

namespace DreamHopper
{
    [Guid("8E5163A1-DB3D-4722-908A-361605B77AD5")]
    public class DreamHopperHost : RhinoWindows.Controls.WpfElementHost
    {
        public DreamHopperHost(RhinoDoc docSn) : base(new MainView(docSn), null) {  }
    }
}

