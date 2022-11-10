using System;

namespace DreamHopper.ViewModels
{
    public class CaptureImageEventArgs : EventArgs
    {
        public CaptureImageEventArgs(int width, int height)
        {
            this.Width = width;
            this.Height = height;
        }

        public int Width { get; set; }
        public int Height { get; set; }
    }
}

