using MaterialDesignThemes.Wpf;
using System;
using System.Windows.Controls;
using System.Windows.Media;
using System.Windows.Shapes;

namespace DreamHopper.ViewModels.Helpers
{
    public static class SnackBarContentCreator
    {
        private static Color SuccessColor => Color.FromRgb(117, 207, 15);
        private static Color InfoColor => Color.FromRgb(43, 50, 59);
        private static Color WarningColor => Color.FromRgb(255, 183, 10);
        private static Color ErrorColor => Color.FromRgb(243, 57, 81);

        [STAThread]
        public static Grid CreateSuccessMessage(string msg)
        {
            PackIcon icon = CreatePackIcon(25, PackIconKind.Check);
            return CreateGrid(icon, new Label() { Content = msg, Foreground = Brushes.White, HorizontalAlignment = System.Windows.HorizontalAlignment.Center }, new SolidColorBrush(SuccessColor), false);
        }

        [STAThread]
        public static Grid CreateInfoMessage(string msg)
        {
            PackIcon icon = CreatePackIcon(25, PackIconKind.Information);
            return CreateGrid(icon, new Label() { Content = msg, Foreground = Brushes.White, HorizontalAlignment = System.Windows.HorizontalAlignment.Center }, new SolidColorBrush(InfoColor), false);
        }

        [STAThread]
        public static Grid CreateWarningMessage(string msg)
        {
            PackIcon icon = CreatePackIcon(25, PackIconKind.Warning);
            return CreateGrid(icon, new Label() { Content = msg, Foreground = Brushes.White, HorizontalAlignment = System.Windows.HorizontalAlignment.Center }, new SolidColorBrush(WarningColor));
        }

        [STAThread]
        public static Grid CreateErrorMessage(string msg)
        {
            PackIcon icon = CreatePackIcon(25, PackIconKind.Error);
            return CreateGrid(icon, new Label() { Content = msg, Foreground = Brushes.White, HorizontalAlignment = System.Windows.HorizontalAlignment.Center }, new SolidColorBrush(ErrorColor));
        }

        [STAThread]
        public static PackIcon CreatePackIcon(double dim, PackIconKind kind)
        {

            PackIcon icon = new PackIcon();
            icon.Foreground = Brushes.White;
            icon.Width = dim;
            icon.Height = dim;
            icon.Kind = kind;
            return icon;
        }

        [STAThread]
        public static Grid CreateGrid(PackIcon icon, Label msg, Brush gridColor, bool includeCopyButton = true)
        {
            Rectangle rect = new Rectangle();
            rect.Fill = gridColor;
            rect.RadiusX = 4;
            rect.RadiusY = 4;
            rect.Margin = new System.Windows.Thickness(-5);

            ColumnDefinition iconColumn = new ColumnDefinition();
            iconColumn.Width = new System.Windows.GridLength(25);

            ColumnDefinition separatorColumnOne = new ColumnDefinition();
            separatorColumnOne.Width = new System.Windows.GridLength(10);

            ColumnDefinition textColumn = new ColumnDefinition();

            ColumnDefinition separatorColumnTwo = new ColumnDefinition();
            separatorColumnTwo.Width = new System.Windows.GridLength(10);

            ColumnDefinition expanderColumn = new ColumnDefinition();
            expanderColumn.Width = new System.Windows.GridLength(25);

            Grid grd = new Grid();
            grd.Children.Add(rect);
            grd.ColumnDefinitions.Add(iconColumn);
            grd.ColumnDefinitions.Add(separatorColumnOne);
            grd.ColumnDefinitions.Add(textColumn);

            if (includeCopyButton)
            {
                grd.ColumnDefinitions.Add(separatorColumnTwo);
                grd.ColumnDefinitions.Add(expanderColumn);

                Button button = new Button();
                button.Content = CreatePackIcon(20, PackIconKind.ContentDuplicate);
                button.Background = gridColor;
                button.BorderBrush = Brushes.Transparent;
                button.BorderThickness = new System.Windows.Thickness(0);
                button.Padding = new System.Windows.Thickness(0);
                button.Click += Button_Click;
                button.DataContext = msg.Content;

                grd.Children.Add(button);
                Grid.SetColumn(button, 4);
            }

            grd.ToolTip = msg.Content;



            grd.Children.Add(icon);
            Grid.SetColumn(icon, 0);

            grd.Children.Add(msg);
            Grid.SetColumn(msg, 2);

            Grid.SetColumnSpan(rect, 5);


            grd.HorizontalAlignment = System.Windows.HorizontalAlignment.Center;

            return grd;
        }

        private static void Button_Click(object sender, System.Windows.RoutedEventArgs e)
        {
            Button b = sender as Button;
            string msg = b.DataContext as string;
            if (msg != null)
            {
                System.Windows.Forms.Clipboard.SetText(msg);
            }
        }
    }
}
