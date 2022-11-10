using DreamHopper.ViewModels.ViewModels;
using Rhino;
using Rhino.Commands;
using Rhino.Geometry;
using Rhino.Input;
using Rhino.Input.Custom;
using Rhino.UI;
using System;
using System.Collections.Generic;

namespace DreamHopper
{
    public class DreamHopperCommand : Command
    {
        public DreamHopperCommand()
        {
            Instance = this;
            Panels.RegisterPanel(DreamHopperPlugin.Instance, typeof(DreamHopperHost), "DreamHopper", Properties.Resources.DreamHopperIcon);

        }

        ///<summary>The only instance of this command.</summary>
        public static DreamHopperCommand Instance { get; private set; }

        ///<returns>The command name as it appears on the Rhino command line.</returns>
        public override string EnglishName => "DreamHopper";

        protected override Result RunCommand(RhinoDoc doc, RunMode mode)
        {
            RhinoApp.WriteLine("Launching DreamHopper... Yasssss!");
            this.DreamHopperLauncher();
            return Result.Success;
        }
        private void DreamHopperLauncher()
        {
            var panel_id = typeof(DreamHopperHost).GUID;
            var panel_visible = Panels.IsPanelVisible(panel_id);
            if (!panel_visible)
                Panels.OpenPanel(panel_id);
        }
    }
}
