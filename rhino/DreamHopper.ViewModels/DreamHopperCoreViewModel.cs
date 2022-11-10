using DreamHopper.IO;
using DreamHopper.MVVM;
using DreamHopper.ViewModels.Helpers;
using DreamHopper.ViewModels.ViewModels;
using DreamHopper.WebClient;
using MaterialDesignThemes.Wpf;
using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Drawing;
using System.Drawing.Imaging;
using System.IO;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using System.Windows.Data;
using System.Windows.Forms;
using System.Windows.Input;

namespace DreamHopper.ViewModels
{
    public class DreamHopperCoreViewModel : BaseViewModel
    {
        public static DreamHopperCoreViewModel Instance;
        public string DocName { get; private set; }
        public bool OutputsExpanded { get; set; }
        public bool Processing { get; set; }

        public ICommand BakeCommand { get; set; }
        public ICommand RunInferenceCommand { get; set; }

        public bool CanBake
        {
            get { return this.Mesh != null && this.Mesh.IsValid && !this.Processing; }
            set { }
        }


        private InferenceClient _client;
        private SubmissionReceipt _receipt;
        public SnackbarMessageQueue MessageQueue { get; set; }
        public Bitmap ReferenceImage { get; set; }
        public bool HasReferenceImage { get { return ReferenceImage != null; } set { } }

        public ObservableCollection<BaseInputViewModel> Inputs { get; set; }
        public DreamHopperMesh Mesh { get; set; }

        private object _outputsLock = new object();

        public DreamHopperCoreViewModel(string docName)
        {
            Instance = this;
            this.DocName = docName;
            //this.CaptureImageCommand = new RelayCommand(this.CaptureImage);
            //this.OpenImageCommand = new RelayCommand(this.OpenImage);
            this.RunInferenceCommand = new RelayCommand(async () => await this.RunInference());
            this.BakeCommand = new RelayCommand(this.Bake);
            this._client = new InferenceClient();
            this.MessageQueue = new SnackbarMessageQueue(new TimeSpan(0, 0, 3));
            this.Processing = false;

            this.Inputs = new ObservableCollection<BaseInputViewModel>()
            {
                new TextInputViewModel("Prompt:", "text"),
                new NumberSliderInputViewModel("Iterations:", "iters", 0, 50000, 2000, 1),
                new NumberSliderInputViewModel("Clip Size:", "clip", 128, 768, 160),
                new NumberSliderInputViewModel("Threshold:", "threshold", 0, 50, 10, 0.1),
                new IntegerInputViewModel("Seed:", "seed", -1)
            };

            this.PropertyChanged += DreamHopperCoreViewModel_PropertyChanged;
        }


        private void DreamHopperCoreViewModel_PropertyChanged(object sender, System.ComponentModel.PropertyChangedEventArgs e)
        {
            if (e.PropertyName == nameof(this.Mesh) || e.PropertyName == nameof(this.Processing))
            {
                this.RaisePropertyChanged(nameof(this.CanBake));
            }
        }

        public void CaptureImage()
        {
            CaptureImageEventArgs args = new CaptureImageEventArgs(512, 512);
            this.OnImageCaptureTriggered(args);
        }

        public async Task RunInference()
        {
            Dictionary<string, object> dict = new Dictionary<string, object>();

            foreach (BaseInputViewModel vm in this.Inputs)
            {
                KeyValuePair<string, object> entry = vm.GetValuePair();
                dict.Add(entry.Key, entry.Value);
            }

            //string inputImage = this.InputImageAsBase64();
            string prompt = dict["text"] as string ?? "";
            string user = Environment.UserName;
            int iterations = (int)((double)dict["iters"]);
            int clip = (int)((double)dict["clip"]);
            int seed = (int)dict["seed"];
            double threshold = (double)dict["threshold"];

            //Communicate the server
            DreamHopperDTO dto = new DreamHopperDTO(
                prompt,
                user,
                iterations,
                clip,
                threshold,
                seed);

            try
            {
                _receipt = await this._client.SubmitRequest(dto);
            }
            catch (Exception exe)
            {
                this.MessageQueue.Enqueue(SnackBarContentCreator.CreateErrorMessage(exe.Message));
                this.Processing = false;
            }

            if (_receipt != null)
            {
                Task.Run(async () =>
                {
                    this.Processing = true;
                    this.OutputsExpanded = true;
                    bool isDone = false;
                    DreamHopperDTO response = new DreamHopperDTO();

                    int errorCounter = 0;

                    while (!isDone)
                    {
                        try
                        {
                            response = await this._client.CheckRequestStatus(this._receipt);
                            isDone = response.Done;
                            Thread.Sleep(this._client.CheckFrequency);
                        }
                        catch (Exception e)
                        {
                            this.MessageQueue.Enqueue(SnackBarContentCreator.CreateErrorMessage(e.Message));
                            errorCounter += 1;
                            if (errorCounter > 10) isDone = true;
                        }
                    }
                    this.Processing = false;

                    try
                    {
                        this.Mesh = response.Mesh;
                        // 
                    }
                    catch (Exception exc)
                    {
                        this.MessageQueue.Enqueue(SnackBarContentCreator.CreateErrorMessage(exc.Message));
                    }


                });
            }


        }

        public void Bake()
        {
            Rhino.Geometry.Mesh mesh = this.Mesh.CreateMesh();
            Rhino.RhinoDoc.ActiveDoc.Objects.AddMesh(mesh);
            Rhino.RhinoDoc.ActiveDoc.Views.Redraw();
        }

        public EventHandler ImageCaptureTriggered;
        public void OnImageCaptureTriggered(CaptureImageEventArgs e)
        {
            EventHandler handler = ImageCaptureTriggered;
            handler?.Invoke(this, e);
        }
    }
}