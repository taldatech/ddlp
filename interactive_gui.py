import tkinter as tk
from tkinter import ttk
from ttkthemes import ThemedTk
from ttkwidgets import TickScale
from PIL import Image, ImageTk
import threading
import numpy as np
import os
import json
from tqdm import tqdm

import torch
from torchvision.transforms import ToTensor, ToPILImage
from models import ObjectDynamicsDLP, ObjectDLP
from modules.diffusion_modules import PINTDenoiser, GaussianDiffusionPINT, TrainerDiffuseDDLP
from train_diffuse_ddlp import ParticleNormalization

# COLORS = ['red', 'blue', 'yellow', 'green', 'purple', 'orange']
COLORS = ['#FF0000', '#0000FF', '#FFFF00', '#00FF00', '#800080', '#FFA500',
          '#FFC0CB', '#008000', '#FF69B4', '#800000', '#FFD700', '#000080',
          '#808080', '#FF7F50', '#FFA07A', '#FF4500', '#00FFFF', '#00008B',
          '#FF1493', '#008080', '#8B0000', '#7B68EE', '#228B22', '#20B2AA',
          '#DAA520', '#4B0082', '#FF00FF', '#00FF7F', '#1E90FF', '#B22222',
          '#C71585', '#ADFF2F', '#F0E68C', '#FF8C00', '#FFDAB9', '#FA8072',
          '#FFFFE0', '#00BFFF', '#2E8B57', '#FAEBD7', '#FF4500', '#D8BFD8']


class KeyPoint:
    def __init__(self, canvas, x, y, scale, index, feature_index, features=(None,), scale_multiplier=1.0, obj_on=1.0,
                 glimpse=None, timestep=0, to_kp=None, gui=None):
        self.canvas = canvas
        self.x = x
        self.y = y
        self.scale = scale  # [2, ]
        self.index = index
        self.feature_index = feature_index
        self.features = features[feature_index]
        self.features_dict = {f'{i}': features[i] for i in range(len(features))}
        self.timestep = timestep
        self.to_kp = to_kp  # (x, y)
        self.gui = gui
        self.radius = 5 if (timestep == 0 or timestep == self.gui.n_frames - 1) else 2
        self.scale_multiplier = scale_multiplier
        self.obj_on = obj_on
        self.glimpse = glimpse
        self.glimpse_size = 100

        text = f"{self.index}[t={self.timestep}]" if self.timestep == 0 else f''
        self.text = canvas.create_text(
            x + 10,
            y + 10,
            text=text,
            anchor=tk.NW,
            fill='black',
            tags='keypoint'
        )
        self.item = canvas.create_oval(
            x - self.radius,
            y - self.radius,
            x + self.radius,
            y + self.radius,
            fill=COLORS[index % len(COLORS)],
            tags='keypoint'
        )

        if self.to_kp is not None:
            self.arrow = self.canvas.create_line(x, y, self.to_kp[0], self.to_kp[1], arrow=tk.LAST,
                                                 fill=COLORS[index % len(COLORS)],
                                                 tags='keypoint')

        self.canvas.tag_bind(self.item, '<ButtonPress-1>', self.on_press)
        self.canvas.tag_bind(self.item, '<B1-Motion>', self.on_drag)
        self.canvas.tag_bind(self.item, '<ButtonRelease-1>', self.on_release)

        self.kp_label = None
        self.slider, self.slider_label = None, None
        self.scroller, self.scroller_label = None, None
        self.slider_obj, self.slider_obj_label = None, None
        self.gcanvas = None
        self.img_tk = None
        self.selected = False

    def select_kp(self):
        self.canvas.itemconfigure(self.item, width=3, outline='black')
        self.selected = True
        self.gui.selected_keypoints.append(self)

    def deselect_kp(self):
        self.selected = False
        self.canvas.itemconfigure(self.item, width=1, outline='')

    def on_press(self, event):
        self.start_x = event.x
        self.start_y = event.y
        for kp in self.gui.selected_keypoints:
            kp.start_x = event.x
            kp.start_y = event.y
        if self.gui is not None:
            gui.close_all()
        if self.kp_label is not None:
            self.kp_label.destroy()
        if self.slider is not None:
            self.slider.destroy()
            self.slider_label.destroy()
        if self.slider_obj is not None:
            self.slider_obj.destroy()
            self.slider_obj_label.destroy()
        if self.scroller is not None:
            self.scroller.destroy()
            self.scroller_label.destroy()
        if self.gcanvas is not None:
            self.gcanvas.destroy()
            self.img_tk = None
        if not self.selected:
            self.select_kp()
            # Create canvas to display image
            self.gcanvas = tk.Canvas(root, width=self.glimpse_size, height=self.glimpse_size)
            # self.canvas.pack()
            self.gcanvas.grid(row=6, column=1)
            self.img_tk = ImageTk.PhotoImage(
                self.glimpse.resize((self.glimpse_size, self.glimpse_size), Image.LANCZOS))
            self.gcanvas.create_image(0, 0, anchor=tk.NW, image=self.img_tk)

            self.kp_label = ttk.Label(
                root,
                text=f'particle #{self.index} [t={self.timestep}]', font=('Arial', 10), background='#818485',
                foreground='black'
            )
            # self.kp_label.pack(side=tk.TOP)
            self.kp_label.grid(row=6, column=0)
            if self.timestep == 0:
                self.slider_label = ttk.Label(
                    root,
                    text='scale:', font=('Arial', 10), background='#818485', foreground='black'
                )
                # self.slider_label.pack(side=tk.TOP)
                self.slider_label.grid(row=7, column=1)
                self.slider = TickScale(
                    root,
                    from_=0.1,
                    to=3.0,
                    orient=tk.HORIZONTAL,
                    length=150,
                    # resolution=0.1,
                    # showvalue=True,
                    command=self.on_scale_change
                )

                self.slider.set(self.scale_multiplier)
                # self.slider.pack(side=tk.TOP)
                self.slider.grid(row=8, column=1)

                self.slider_obj_label = ttk.Label(
                    root,
                    text='transparency:', font=('Arial', 10), background='#818485', foreground='black'
                )
                # self.slider_obj_label.pack(side=tk.TOP)
                self.slider_obj_label.grid(row=9, column=1)

                self.slider_obj = TickScale(
                    root,
                    from_=0.0,
                    to=1.0,
                    orient=tk.HORIZONTAL,
                    length=150,
                    # resolution=0.1,
                    # showvalue=True,
                    command=self.on_obj_on_change
                )
                self.slider_obj.set(self.obj_on)
                # self.slider_obj.pack(side=tk.TOP)
                self.slider_obj.grid(row=10, column=1)

                # Create and place the features menu
                self.scroller_label = ttk.Label(
                    root,
                    text='visual appearance:', font=('Arial', 10), background='#818485', foreground='black'
                )
                # self.scroller_label.pack(side=tk.TOP)
                self.scroller_label.grid(row=11, column=1)

                feat_var = tk.StringVar()
                feat_var.set(f'{self.feature_index}')  # Set the default value
                self.scroller = ttk.OptionMenu(
                    root,
                    feat_var,
                    f'{self.feature_index}',  # Set the default values
                    *list(self.features_dict.keys()), command=self.on_features_change, style='my.TMenubutton'
                )
                self.scroller['menu'].configure(font=('Arial', 10))
                # self.scroller.pack(side=tk.TOP, pady=10)
                self.scroller.grid(row=12, column=1)

    def on_drag(self, event):
        if self.kp_label is not None:
            self.kp_label.destroy()
        if self.slider is not None:
            self.slider.destroy()
            self.slider_label.destroy()
        if self.slider_obj is not None:
            self.slider_obj.destroy()
            self.slider_obj_label.destroy()
        if self.scroller is not None:
            self.scroller.destroy()
            self.scroller_label.destroy()
        if self.gcanvas is not None:
            self.gcanvas.destroy()
            self.img_tk = None
        # if self.selected:

        for kp in self.gui.selected_keypoints:
            if kp.timestep == 0 or kp.timestep == self.gui.n_frames - 1:
                dx = event.x - kp.start_x
                dy = event.y - kp.start_y
                self.canvas.move(kp.item, dx, dy)
                self.canvas.move(kp.text, dx, dy)
                if kp.to_kp is not None:
                    self.canvas.move(kp.arrow, dx, dy)
                kp.start_x = event.x
                kp.start_y = event.y

        # dx = event.x - self.start_x
        # dy = event.y - self.start_y
        # self.canvas.move(self.item, dx, dy)
        # self.canvas.move(self.text, dx, dy)
        # self.start_x = event.x
        # self.start_y = event.y

    def on_release(self, event):
        self.deselect_kp()

    def get_coordinates(self):
        coords = self.canvas.coords(self.item)  # [4,]
        x = coords[0] + self.radius
        y = coords[1] + self.radius
        kp = np.array([x, y])
        return kp

    def on_scale_change(self, scale):
        self.scale_multiplier = float(scale)
        # self.canvas.itemconfigure(self.text, text=f"{self.index}[t={self.timestep}]")
        for kp in self.gui.selected_keypoints:
            kp.scale_multiplier = float(scale)
            # self.canvas.itemconfigure(kp.text, text=f"{kp.index}[t={self.timestep}]")
            print(f'{kp.index}: scale-multiplier: {kp.scale_multiplier}')

    def on_obj_on_change(self, obj_on):
        self.obj_on = float(obj_on)
        for kp in self.gui.selected_keypoints:
            kp.obj_on = float(obj_on)
            print(f'{kp.index}: obj_on: {kp.obj_on}')

    def on_features_change(self, index):
        self.feature_index = int(index)
        self.features = self.features_dict[f'{index}']
        for kp in self.gui.selected_keypoints:
            kp.feature_index = int(index)
            kp.features = kp.features_dict[f'{index}']
            print(f'{kp.index}: features: {kp.features}')

    def get_scale(self):
        return self.scale * (1 / self.scale_multiplier)

    def get_scale_multiplier(self):
        return self.scale_multiplier

    def get_obj_on(self):
        return self.obj_on

    def get_features(self):
        return self.features

    def get_features_index(self):
        return self.feature_index


class GUI:
    def __init__(self, root, canvas_size=500):
        self.root = root
        self.root.title("(D)DLP GUI")
        self.canvas_size = canvas_size
        self.canvas = None
        s = ttk.Style()
        s.configure('my.TButton', font=('Arial', 10))
        s.configure('my.TMenubutton', font=('Arial', 10))
        os.makedirs('./checkpoints', exist_ok=True)
        self.available_models = [d for d in os.listdir('./checkpoints') if
                                 ('dlp' in d and os.path.isdir(f'./checkpoints/{d}') and len(os.listdir(f'./checkpoints/{d}')) > 0)]
        if len(self.available_models) == 0:
            raise SystemExit(f'Pre-trained models not found.'
                             f' Please download and place them in the "checkpoints" directory under the root directory.'
                             f' For example: root -> checkpoints > dlp-obj3d128.'
                             f' The directory should include 2 files: .pth chekpoint file and .json config file')
        print(f'available models: {self.available_models}')
        # Create and place the models menu
        self.scroller_label = ttk.Label(
            root,
            text='model:', font=('Arial', 10), background='#818485', foreground='black'
        )
        # self.scroller_label.pack(side=tk.TOP)
        self.scroller_label.grid(row=0, column=0)

        feat_var = tk.StringVar()
        feat_var.set(f'')  # Set the default value
        self.scroller = ttk.OptionMenu(
            root,
            feat_var,
            f'',  # Set the default values
            *list([f''] + self.available_models), command=self.on_model_select, style='my.TMenubutton'
        )
        self.scroller['menu'].configure(font=('Arial', 10))
        # self.scroller.pack(side=tk.TOP, pady=10)
        self.scroller.grid(row=1, column=0, pady=10)

        # Create and place the device menu
        # self.scroller_device_label = ttk.Label(
        #     root,
        #     text='device:', font=('Arial', 10), background='#818485', foreground='black'
        # )
        # # self.scroller_device_label.pack(side=tk.TOP)
        # self.scroller_device_label.grid(row=1, column=3)

        device_var = tk.StringVar()
        device_var.set(f'cpu')  # Set the default value
        self.available_devices = ['cpu', 'cuda:0'] if torch.cuda.is_available() else ['cpu']
        self.device_name = 'cpu'
        self.scroller_device = ttk.OptionMenu(
            root,
            device_var,
            f'cpu',  # Set the default values
            *self.available_devices, command=self.on_device_select, style='my.TMenubutton'
        )
        self.scroller_device['menu'].configure(font=('Arial', 10))
        # self.scroller_device.pack(side=tk.TOP, pady=10)
        self.scroller_device.grid(row=1, column=3, pady=10)

        # self.image_path = image_path
        self.scroller_example_label = None
        self.scroller_example = None
        self.anim_cbox_label = None
        self.anim_cbox = None
        self.hide_kp_label = None
        self.hide_kp = None
        self.gen_btn = None

        self.ds_name = None
        self.example_dir = None
        self.available_examples = None
        self.chosen_example = None
        self.image_path = None
        self.img_container = None
        self.img = None
        self.diffusion_img = None
        self.img_tk = None
        self.seq_path = None
        self.seq_img = None

        self.coordinates = self.original_keypoints = None
        self.scales = self.original_scales = None
        self.depths = self.original_depths = None
        self.original_scale_multiplires = None
        self.features = self.original_features = None
        self.features = self.original_features = None
        self.original_feature_indices = None
        self.obj_ons = self.original_obj_ons = None
        self.original_bg = None

        self.keypoints = []
        self.selected_keypoints = []
        self.selection_rect = None
        self.selection_start = None
        self.n_kp = 0
        self.n_frames = 1
        self.num_interpolation_steps = 10

        self.update_btn = None
        self.reset_btn = None
        self.play_btn = None
        self.model = None
        self.model_type = None
        self.model_name = ''
        self.diffusion_model = None
        self.animate_transitions = tk.BooleanVar(value=False)
        self.hide_particles = tk.BooleanVar(value=False)

        self.quit_btn = ttk.Button(root, text="Exit", command=self.quit, style='my.TButton')
        # self.quit_btn.pack(side=tk.TOP, padx=10, pady=10, ipadx=10, ipady=10)
        self.quit_btn.grid(row=0, column=3, padx=10, pady=10, ipadx=10, ipady=10)

        self.help_btn = ttk.Button(root, text="Help", command=self.view_instructions, style='my.TButton')
        self.help_btn.grid(row=5, column=3, padx=10, pady=10, ipadx=10, ipady=10)

        self.instructions_str = "(D)DLP GUI Usage Instructions: \n" \
                                "*** IMPORTANT NOTE ***:" \
                                " DDLP only works with small latent modifications, and cannot handle modifications that" \
                                " result in out-of-distribution examples.\n" \
                                "1. Choosing a pre-trained model: \n The GUI looks for models inside the `checkpoints`" \
                                " directory. \n The GUI supports 3 types of models: [`dlp`, `ddlp`, `diffuse-ddlp`]," \
                                " and the pre-trained paths should be organized as follows:\n" \
                                " `checkpoints/{model-type}-{ds}/[hparams.json, {ds}_{model-type}.pth]` for `dlp`/`ddlp`" \
                                " and \n `checkpoints/diffuse-ddlp-{ds}/[ddlp_hparams.json," \
                                " diffusion_hparams.json, {ds}_ddlp.pth, latent_stats.pth, /saves/model.pth]` for" \
                                " `diffuse-ddlp`. \n For example: `checkpoints/ddlp-obj3d128/[hparams.json," \
                                " obj3d128_ddlp.pth]`. \n\n" \
                                "2. Choosing/generating an example: \n For `dlp/ddlp` " \
                                "the GUI looks for examples for a dataset in the `assets` directory, where each " \
                                "example is a directory with an integer number as its name." \
                                " \n Under each example directory there" \
                                " should be images, where for `ddlp` at least 4 consecutive are required," \
                                " numbered by their order. \n For example, `assets/obj3d128/428/[1.png, 2.png, ...]`. \n" \
                                " For `diffuse-ddlp`, press the `Generate button` to generate a sequence of 4 (latent)" \
                                " frames. \n\n" \
                                "3. Choosing a device: \n the GUI can run everything on the CPU, but if CUDA " \
                                "is available, you can switch to a GPU to perform computations. \n\n" \
                                "4. Animating latent transitions: \n if the `animate` checkbox is marked, the GUI" \
                                " will animate the latent interpolation between modifications" \
                                " after the `Update` button is pressed (naturally, this is slower). \n\n" \
                                "5. Hiding particles: \n you can temporarily hide the particles to view the current image " \
                                "by marking the `hide particles` checkbox. Removing the check will restore the particles. \n\n" \
                                "6. Latent modifications:\n " \
                                "The GUI supports the following modifications: \n " \
                                " * Moving particles by dragging. Use the selection tool to select multiple particles at" \
                                " once and then drag them all together. This is useful when objects are assigned multiple" \
                                " particles.\n " \
                                " * Changing scale/transparency: once a particle is pressed on, a modifications menu will " \
                                "open.\n   You can change the scale and transparency for multiple particles at once by first " \
                                "pressing a particle, \n   and then using the selection tool to select multiple particles and " \
                                "all changes applied to the pressed particle will be applied to all of them.\n  " \
                                "* Changing visual appearance: when an example is selected/generated,\n  " \
                                " the visual features of all particles are saved in a dictionary where the key " \
                                "is the particle number. You can choose between these available features. \n  " \
                                " Similarly, you can change the features of multiple features at once by first pressing on " \
                                " a particle and" \
                                "\n   then use the selection tool to pick all the particles that will be changed.\n\n" \
                                "7. Video prediction:\n  When using a `ddlp`-based model, you can unroll the latent particles \n" \
                                "  and generate a video in a new window by pressing the `Play` button.\n  " \
                                "Note for DDLP, to make things simpler, we only allow changes to particles at t=0 and t=3\n " \
                                " and interpolate all particles in-between. \n  " \
                                "Note that DDLP is quite sensitive to out-of-distribution modifications."
        print(self.instructions_str)

    def quit(self):
        exit(0)

    def clear_screen(self):
        self.clear_buttons()
        self.image_path = None
        self.selection_rect = None
        self.selection_start = None
        if self.scroller_example_label is not None:
            self.scroller_example_label.destroy()
        if self.scroller_example is not None:
            self.scroller_example.destroy()
        if self.gen_btn is not None:
            self.gen_btn.destroy()
        if self.anim_cbox_label is not None:
            self.anim_cbox_label.destroy()
        if self.anim_cbox is not None:
            self.anim_cbox.destroy()
        if self.hide_kp_label is not None:
            self.hide_kp_label.destroy()
        if self.hide_kp is not None:
            self.hide_kp.destroy()
        if self.model is not None:
            self.model.to(torch.device('cpu'))
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                print('empty cuda cache')
            del self.model
            self.model = None
        if self.canvas is not None:
            self.close_all()
            # self.reset_image()
            # self.canvas.delete('all')
            self.canvas.destroy()

    def deselect_all_keypoints(self):
        for keypoint in self.selected_keypoints:
            keypoint.deselect_kp()
        self.selected_keypoints = []

    def on_device_select(self, device_name):
        self.device_name = device_name
        if self.model is not None:
            self.model.to(torch.device(self.device_name))
        print(f'current device: {self.device_name}')

    def on_model_select(self, dir_name):
        # clear
        self.clear_screen()
        self.model_name = dir_name
        if dir_name != '':
            # self.model_type = 'ddlp' if 'ddlp' in dir_name else 'dlp'
            if 'diffuse' in dir_name:
                self.model_type = 'diffuse_ddlp'
            elif 'ddlp' in dir_name:
                self.model_type = 'ddlp'
            else:
                self.model_type = 'dlp'

            if self.model_type == 'ddlp' or self.model_type == 'diffuse_ddlp':
                self.n_frames = 4
            else:
                self.n_frames = 1
            self.load_model()

            if self.model_type != 'diffuse_ddlp':
                # locate example
                self.example_dir = f'./assets/{self.ds_name}'
                # self.example_dir = f'/media/newhd/data/obj3d/train'
                if not os.path.exists(self.example_dir):
                    raise SystemExit(f'Examples for dataset {self.ds_name} not found.'
                                     f' Please make sure to put each example in its own dir under {self.example_dir}.'
                                     f' For example: root -> assets > {self.example_dir} -> 1 -> *.png'
                                     f' The directory should include image files.')
                self.available_examples = os.listdir(self.example_dir)
                if len(self.available_examples) == 0:
                    raise SystemExit(f'Examples for dataset {self.ds_name} not found.'
                                     f' Please make sure to put each example in its own dir under {self.example_dir}.'
                                     f' For example: root -> assets > {self.example_dir} -> 1 -> *.png'
                                     f' The directory should include image files.')
                print(f'available examples: {self.available_examples}')

                # example scroller
                self.scroller_example_label = ttk.Label(
                    root,
                    text='example:', font=('Arial', 10), background='#818485', foreground='black'
                )
                # self.scroller_label.pack(side=tk.TOP)
                self.scroller_example_label.grid(row=0, column=1)

                example_var = tk.StringVar()
                example_var.set(f'{self.available_examples[0]}')  # Set the default value
                self.scroller_example = ttk.OptionMenu(
                    root,
                    example_var,
                    f'{self.available_examples[0]}',  # Set the default values
                    *self.available_examples, command=self.on_example_select, style='my.TMenubutton'
                )
                self.scroller_example['menu'].configure(font=('Arial', 10))
                # self.scroller.pack(side=tk.TOP, pady=10)
                self.scroller_example.grid(row=1, column=1, pady=10)
            else:
                # diffuse-ddlp
                # generate button
                self.gen_btn = ttk.Button(root, text="Generate", command=self.on_generate_press, style='my.TButton')
                self.gen_btn.grid(row=1, column=1, padx=10, pady=1, ipadx=2, ipady=1)

            self.anim_cbox_label = ttk.Label(
                root,
                text='   animate:   ', font=('Arial', 10), background='#818485', foreground='black'
            )
            # self.scroller_label.pack(side=tk.TOP)
            self.animate_transitions = tk.BooleanVar(value=False)
            self.anim_cbox_label.grid(row=0, column=2)
            self.anim_cbox = ttk.Checkbutton(root, variable=self.animate_transitions, command=self.on_anim_checkbox)
            self.anim_cbox.grid(row=1, column=2)

    def on_anim_checkbox(self, ):
        print(f'transitions animation: {self.animate_transitions.get()}')

    def get_update_img_func(self):
        if self.animate_transitions.get():
            return self.update_image_threaded()
        else:
            return self.update_image()

    def on_example_select(self, example_dir):
        # clear screen
        self.clear_buttons()
        self.selection_rect = None
        self.selection_start = None
        self.image_path = None
        self.img = None
        if self.canvas is not None:
            self.close_all()
            # self.canvas.delete('all')
            self.canvas.destroy()
        self.chosen_example = example_dir
        img_dir = os.listdir(os.path.join(self.example_dir, self.chosen_example))
        if len(img_dir) == 0:
            raise SystemExit(f'Examples for dataset {self.ds_name} not found.'
                             f' Please make sure to put each example in its own dir under {self.example_dir}.'
                             f' For example: root -> assets > {self.example_dir} -> {self.chosen_example} -> *_[#].png'
                             f' Where # is an integer number.')
        available_images = list(sorted(img_dir, key=lambda x: int(x.split('.')[-2].split('_')[-1])))
        print(f'available images: {available_images}')
        if len(available_images) < self.n_frames:
            raise SystemExit(f'Not enough frames for DDLP.'
                             f' Required: {self.n_frames}, available: {len(available_images)}.')
        # load image
        self.image_path = os.path.join(self.example_dir, self.chosen_example, available_images[0])
        self.seq_path = [os.path.join(self.example_dir, self.chosen_example, available_images[t]) for t in
                         range(self.n_frames)]
        # Create buttons
        self.update_btn = ttk.Button(root, text="Update", command=self.get_update_img_func, style='my.TButton')
        # self.update_btn.pack(side=tk.BOTTOM, padx=10, pady=10, ipadx=10, ipady=10)
        self.update_btn.grid(row=5, column=1, padx=10, pady=10, ipadx=10, ipady=10)

        self.reset_btn = ttk.Button(root, text="Reset", command=self.reset_image, style='my.TButton')
        # self.reset_btn.pack(side=tk.TOP, padx=10, pady=10, ipadx=10, ipady=10)
        self.reset_btn.grid(row=3, column=0, padx=10, pady=10, ipadx=10, ipady=10)

        if 'ddlp' in self.model_type:
            self.play_btn = ttk.Button(root, text="Play", command=self.play_video, style='my.TButton')
            # self.play_btn.pack(side=tk.BOTTOM, padx=10, pady=10, ipadx=10, ipady=10)
            self.play_btn.grid(row=3, column=2, padx=10, pady=10, ipadx=10, ipady=10)

        self.hide_particles = tk.BooleanVar(value=False)
        self.hide_kp_label = ttk.Label(
            root,
            text='  hide particles:  ', font=('Arial', 10), background='#818485', foreground='black'
        )
        self.hide_kp_label.grid(row=4, column=0)
        self.hide_kp = ttk.Checkbutton(root, variable=self.hide_particles, command=self.on_hide_kp)
        self.hide_kp.grid(row=5, column=0)

        # Create canvas to display image
        self.canvas = tk.Canvas(root, width=self.canvas_size, height=self.canvas_size)
        # self.canvas.pack()
        self.canvas.grid(row=3, column=1)

        # Load and display the image
        self.load_image()

        # get particles
        particle_dict = self.get_particles()
        if self.model_type == 'dlp':
            kp = particle_dict['z'][0]  # [n_kp, 2], [-1, 1]
            kp = self.normalize_kp(kp, normalize=False)
            kp = kp.cpu().numpy()

            scales = particle_dict['z_scale'][0].cpu().numpy()
            depths = particle_dict['z_depth'][0].cpu().numpy()
            features = particle_dict['z_features'][0].cpu().numpy()
            obj_ons = particle_dict['obj_on'][0].cpu().numpy()
            bg = particle_dict['z_bg'][0].cpu().numpy()

            glimpses = particle_dict['dec_objects_original']
            alpha, rgb = torch.split(glimpses, [1, 3], dim=2)
            rgba = alpha * rgb  # [1, n_particles, 3, h, w]
            rgba = rgba[0].clamp(0, 1).cpu()
            glimpses = [ToPILImage()(rgba[i]) for i in range(rgba.shape[0])]
            self.original_glimpses = glimpses
        else:
            # ddlp: [T, n_kp, features]
            kp = particle_dict['z']  # [T, n_kp, 2], [-1, 1]
            kp = self.normalize_kp(kp, normalize=False)
            kp = kp.permute(1, 0, 2).cpu().numpy()  # [n_kp, T, features]

            scales = particle_dict['z_scale'].permute(1, 0, 2).cpu().numpy()
            depths = particle_dict['z_depth'].permute(1, 0, 2).cpu().numpy()
            features = particle_dict['z_features'].permute(1, 0, 2).cpu().numpy()
            obj_ons = particle_dict['obj_on'].permute(1, 0).cpu().numpy()
            bg = particle_dict['z_bg'].cpu().numpy()  # [T, f]

            glimpses = particle_dict['dec_objects_original']
            alpha, rgb = torch.split(glimpses, [1, 3], dim=2)
            rgba = alpha * rgb  # [T, n_particles, 3, h, w]
            rgba = rgba.clamp(0, 1).permute(1, 0, 2, 3, 4).cpu()  # [n_particles, T, 3, h, w]
            glimpses = []
            for i in range(rgba.shape[0]):
                kp_glimpses = [ToPILImage()(rgba[i, j]) for j in range(rgba.shape[1])]
                glimpses.append(kp_glimpses)
            self.original_glimpses = glimpses

        self.keypoints = []
        self.selected_keypoints = []
        self.coordinates = self.original_keypoints = kp
        self.scales = self.original_scales = scales
        self.original_scale_multiplires = np.ones_like(obj_ons)
        self.features = self.original_features = features
        self.obj_ons = self.original_obj_ons = obj_ons
        self.depths = self.original_depths = depths
        self.original_bg = bg
        # Add keypoints
        if self.model_type == 'dlp':
            self.original_feature_indices = list(range(len(kp)))
            self.add_keypoints(keypoints=kp, scales=scales, scale_multipliers=self.original_scale_multiplires,
                               obj_ons=self.original_obj_ons,
                               features=self.original_features, feature_indices=self.original_feature_indices,
                               glimpses=glimpses)
        else:
            self.original_feature_indices = np.array(list(range(len(kp))))[:, None].repeat(self.n_frames, axis=1)
            self.add_keypoints_trajectory(keypoints=kp, scales=scales,
                                          scale_multipliers=self.original_scale_multiplires,
                                          obj_ons=self.original_obj_ons,
                                          features=self.original_features,
                                          feature_indices=self.original_feature_indices,
                                          glimpses=glimpses)

        self.n_kp = len(self.keypoints)

        self.canvas.bind('<ButtonPress-1>', self.on_canvas_press)
        self.canvas.bind('<B1-Motion>', self.on_canvas_drag)
        self.canvas.bind('<ButtonRelease-1>', self.on_canvas_release)

    def on_hide_kp(self):
        print(f'hide particles: {self.hide_particles.get()}')
        if self.hide_particles.get():
            self.load_image()
        else:
            self.update_image()

    def on_canvas_press(self, event):
        if len(self.selected_keypoints) > 0:
            self.canvas.delete(self.selection_rect)
            self.selection_rect = None
            # self.deselect_all_keypoints()
            # self.selection_rect = None
            # self.selected_keypoints = []
        else:
            self.selection_start = (event.x, event.y)
            self.selection_rect = self.canvas.create_rectangle(event.x, event.y, event.x, event.y, outline='blue')

    def on_canvas_drag(self, event):
        if self.selection_rect is not None:
            x, y = self.selection_start
            self.canvas.coords(self.selection_rect, x, y, event.x, event.y)

    def on_canvas_release(self, event):
        if self.selection_rect is not None:
            x1, y1, x2, y2 = self.canvas.coords(self.selection_rect)
            selected_kp = []
            for kp in self.keypoints:
                px, py, _, _ = self.canvas.coords(kp.item)
                if x1 <= px <= x2 and y1 <= py <= y2:
                    kp.select_kp()
                    selected_kp.append(kp.index)
            print(f'selected particles: {selected_kp}')
            self.canvas.delete(self.selection_rect)
            self.selection_rect = None
        else:
            self.deselect_all_keypoints()

    def load_model(self):
        if self.model_type == 'diffuse_ddlp':
            conf_path = os.path.join('./checkpoints', f'{self.model_name}', 'ddlp_hparams.json')
        else:
            conf_path = os.path.join('./checkpoints', f'{self.model_name}', 'hparams.json')
        with open(conf_path, 'r') as f:
            config = json.load(f)
        ds = config['ds']
        image_size = config['image_size']
        ch = config['ch']
        enc_channels = config['enc_channels']
        prior_channels = config['prior_channels']
        use_correlation_heatmaps = config['use_correlation_heatmaps']
        enable_enc_attn = config['enable_enc_attn']
        filtering_heuristic = config['filtering_heuristic']

        self.ds_name = ds

        if self.model_type == 'ddlp' or self.model_type == 'diffuse_ddlp':
            model_type = 'ddlp'
            self.model = ObjectDynamicsDLP(cdim=ch, enc_channels=enc_channels, prior_channels=prior_channels,
                                           image_size=image_size, n_kp=config['n_kp'],
                                           learned_feature_dim=config['learned_feature_dim'],
                                           pad_mode=config['pad_mode'],
                                           sigma=config['sigma'],
                                           dropout=config['dropout'], patch_size=config['patch_size'],
                                           n_kp_enc=config['n_kp_enc'],
                                           n_kp_prior=config['n_kp_prior'], kp_range=config['kp_range'],
                                           kp_activation=config['kp_activation'],
                                           anchor_s=config['anchor_s'],
                                           use_resblock=config['use_resblock'],
                                           timestep_horizon=config['timestep_horizon'],
                                           predict_delta=config['predict_delta'],
                                           scale_std=config['scale_std'],
                                           offset_std=config['offset_std'], obj_on_alpha=config['obj_on_alpha'],
                                           obj_on_beta=config['obj_on_beta'], pint_heads=config['pint_heads'],
                                           pint_layers=config['pint_layers'], pint_dim=config['pint_dim'],
                                           use_correlation_heatmaps=use_correlation_heatmaps,
                                           enable_enc_attn=enable_enc_attn, filtering_heuristic=filtering_heuristic).to(
                torch.device(f'{self.device_name}'))
        else:
            model_type = 'dlp'
            self.model = ObjectDLP(cdim=ch, enc_channels=enc_channels, prior_channels=prior_channels,
                                   image_size=image_size, n_kp=config['n_kp'],
                                   learned_feature_dim=config['learned_feature_dim'],
                                   pad_mode=config['pad_mode'],
                                   sigma=config['sigma'],
                                   dropout=config['dropout'], patch_size=config['patch_size'],
                                   n_kp_enc=config['n_kp_enc'],
                                   n_kp_prior=config['n_kp_prior'], kp_range=config['kp_range'],
                                   kp_activation=config['kp_activation'],
                                   anchor_s=config['anchor_s'],
                                   use_resblock=config['use_resblock'],
                                   scale_std=config['scale_std'],
                                   offset_std=config['offset_std'], obj_on_alpha=config['obj_on_alpha'],
                                   obj_on_beta=config['obj_on_beta'], use_tracking=config['use_tracking'],
                                   use_correlation_heatmaps=use_correlation_heatmaps,
                                   enable_enc_attn=enable_enc_attn, filtering_heuristic=filtering_heuristic).to(
                torch.device(f'{self.device_name}'))
        model_ckpt_name = os.path.join('./checkpoints', f'{self.model_name}', f'{ds}_{model_type}.pth')
        self.model.load_state_dict(torch.load(model_ckpt_name, map_location=torch.device(f'{self.device_name}')))
        self.model.eval()
        self.model.requires_grad_(False)
        print(f"loaded model from {model_ckpt_name}")

        if self.model_type == 'diffuse_ddlp':
            diff_conf_path = os.path.join('./checkpoints', f'{self.model_name}', 'diffusion_hparams.json')
            with open(diff_conf_path, 'r') as f:
                diffusion_config = json.load(f)
            diffuse_frames = diffusion_config['diffuse_frames']  # number of particle frames to generate
            lr = diffusion_config['lr']
            train_num_steps = diffusion_config['train_num_steps']
            diffusion_num_steps = diffusion_config['diffusion_num_steps']
            loss_type = diffusion_config['loss_type']
            particle_norm = diffusion_config['particle_norm']
            device = torch.device(f'{self.device_name}')
            result_dir = os.path.join('./checkpoints', f'{self.model_name}')
            diffusion_config['result_dir'] = result_dir

            features_dim = 2 + 2 + 1 + 1 + config['learned_feature_dim']
            # features: xy, scale_xy, depth, obj_on, particle features
            # total particles: n_kp + 1 for bg

            denoiser_model = PINTDenoiser(features_dim, hidden_dim=config['pint_dim'],
                                          projection_dim=config['pint_dim'],
                                          n_head=config['pint_heads'], n_layer=config['pint_layers'],
                                          block_size=diffuse_frames, dropout=0.1,
                                          predict_delta=False, positional_bias=True,
                                          max_particles=config['n_kp_enc'] + 1,
                                          self_condition=False,
                                          learned_sinusoidal_cond=False, random_fourier_features=False,
                                          learned_sinusoidal_dim=16).to(device)

            diffusion = GaussianDiffusionPINT(
                denoiser_model,
                seq_length=diffuse_frames,
                timesteps=diffusion_num_steps,  # number of steps
                sampling_timesteps=diffusion_num_steps,
                loss_type=loss_type,  # L1 or L2
                objective='pred_x0',
            ).to(device)

            particle_normalizer = ParticleNormalization(diffusion_config, mode=particle_norm).to(device)

            # expects input: [batch_size, feature_dim, seq_len]

            self.diffusion_model = TrainerDiffuseDDLP(
                diffusion,
                ddlp_model=self.model,
                diffusion_config=diffusion_config,
                particle_norm=particle_normalizer,
                train_batch_size=1,
                train_lr=lr,
                train_num_steps=train_num_steps,  # total training steps
                gradient_accumulate_every=1,  # gradient accumulation steps
                ema_decay=0.995,  # exponential moving average decay
                amp=False,  # turn on mixed precision
                seq_len=diffuse_frames,
                save_and_sample_every=1000,
                results_folder=result_dir
            )

            self.diffusion_model.load()

    def on_generate_press(self):
        particles = torch.randn(1, self.diffusion_model.seq_len,
                                self.diffusion_model.n_particles + 1,
                                self.diffusion_model.ema.ema_model.particle_dim, device=torch.device(self.device_name))
        x_start = None
        for t in tqdm(reversed(range(0, self.diffusion_model.ema.ema_model.num_timesteps)),
                      desc='sampling loop time step', total=self.diffusion_model.ema.ema_model.num_timesteps):
            self_cond = x_start if self.diffusion_model.ema.ema_model.self_condition else None
            particles, x_start = self.diffusion_model.ema.ema_model.p_sample(particles, t, self_cond)

        images, z_dict = self.diffusion_model.latent_to_ddlp_output(particles)

        # clear screen
        self.clear_buttons()
        self.selection_rect = None
        self.selection_start = None
        self.img = None
        if self.canvas is not None:
            self.close_all()
            # self.canvas.delete('all')
            self.canvas.destroy()

        # Create buttons
        self.update_btn = ttk.Button(root, text="Update", command=self.get_update_img_func, style='my.TButton')
        # self.update_btn.pack(side=tk.BOTTOM, padx=10, pady=10, ipadx=10, ipady=10)
        self.update_btn.grid(row=5, column=1, padx=10, pady=10, ipadx=10, ipady=10)

        self.reset_btn = ttk.Button(root, text="Reset", command=self.reset_image, style='my.TButton')
        # self.reset_btn.pack(side=tk.TOP, padx=10, pady=10, ipadx=10, ipady=10)
        self.reset_btn.grid(row=3, column=0, padx=10, pady=10, ipadx=10, ipady=10)

        if 'ddlp' in self.model_type:
            self.play_btn = ttk.Button(root, text="Play", command=self.play_video, style='my.TButton')
            # self.play_btn.pack(side=tk.BOTTOM, padx=10, pady=10, ipadx=10, ipady=10)
            self.play_btn.grid(row=3, column=2, padx=10, pady=10, ipadx=10, ipady=10)

        self.hide_particles = tk.BooleanVar(value=False)
        self.hide_kp_label = ttk.Label(
            root,
            text='  hide particles:  ', font=('Arial', 10), background='#818485', foreground='black'
        )
        self.hide_kp_label.grid(row=4, column=0)
        self.hide_kp = ttk.Checkbutton(root, variable=self.hide_particles, command=self.on_hide_kp)
        self.hide_kp.grid(row=5, column=0)

        # Create canvas to display image
        self.canvas = tk.Canvas(root, width=self.canvas_size, height=self.canvas_size)
        # self.canvas.pack()
        self.canvas.grid(row=3, column=1)

        self.img = ToPILImage()(images[0, 0].clamp(0, 1).cpu())
        self.load_image()

        # get particles
        self.seq_img = images
        particle_dict = self.get_particles()
        # ddlp: [T, n_kp, features]
        kp = particle_dict['z']  # [T, n_kp, 2], [-1, 1]
        kp = self.normalize_kp(kp, normalize=False)
        kp = kp.permute(1, 0, 2).cpu().numpy()  # [n_kp, T, features]

        scales = particle_dict['z_scale'].permute(1, 0, 2).cpu().numpy()
        depths = particle_dict['z_depth'].permute(1, 0, 2).cpu().numpy()
        features = particle_dict['z_features'].permute(1, 0, 2).cpu().numpy()
        obj_ons = particle_dict['obj_on'].permute(1, 0).cpu().numpy()
        bg = particle_dict['z_bg'].cpu().numpy()  # [T, f]

        glimpses = particle_dict['dec_objects_original']
        alpha, rgb = torch.split(glimpses, [1, 3], dim=2)
        rgba = alpha * rgb  # [T, n_particles, 3, h, w]
        rgba = rgba.clamp(0, 1).permute(1, 0, 2, 3, 4).cpu()  # [n_particles, T, 3, h, w]
        glimpses = []
        for i in range(rgba.shape[0]):
            kp_glimpses = [ToPILImage()(rgba[i, j]) for j in range(rgba.shape[1])]
            glimpses.append(kp_glimpses)
        self.original_glimpses = glimpses

        self.keypoints = []
        self.selected_keypoints = []
        self.coordinates = self.original_keypoints = kp
        self.scales = self.original_scales = scales
        self.original_scale_multiplires = np.ones_like(obj_ons)
        self.features = self.original_features = features
        self.obj_ons = self.original_obj_ons = obj_ons
        self.depths = self.original_depths = depths
        self.original_bg = bg
        # Add keypoints
        if self.model_type == 'dlp':
            self.original_feature_indices = list(range(len(kp)))
            self.add_keypoints(keypoints=kp, scales=scales, scale_multipliers=self.original_scale_multiplires,
                               obj_ons=self.original_obj_ons,
                               features=self.original_features, feature_indices=self.original_feature_indices,
                               glimpses=glimpses)
        else:
            self.original_feature_indices = np.array(list(range(len(kp))))[:, None].repeat(self.n_frames, axis=1)
            self.add_keypoints_trajectory(keypoints=kp, scales=scales,
                                          scale_multipliers=self.original_scale_multiplires,
                                          obj_ons=self.original_obj_ons,
                                          features=self.original_features,
                                          feature_indices=self.original_feature_indices,
                                          glimpses=glimpses)

        self.n_kp = len(self.keypoints)

        self.update_image()
        self.diffusion_img = self.img

        self.canvas.bind('<ButtonPress-1>', self.on_canvas_press)
        self.canvas.bind('<B1-Motion>', self.on_canvas_drag)
        self.canvas.bind('<ButtonRelease-1>', self.on_canvas_release)

    def load_image(self):
        if self.img is None:
            if self.diffusion_img is None:
                self.img = Image.open(self.image_path)
                if self.ds_name == 'phyre':
                    self.img = Image.fromarray(255 - np.array(self.img))
            else:
                self.img = self.diffusion_img
        self.img_tk = ImageTk.PhotoImage(self.img.resize((self.canvas_size, self.canvas_size), Image.LANCZOS))
        self.img_container = self.canvas.create_image(0, 0, anchor=tk.NW, image=self.img_tk)
        # self.canvas.create_image(self.canvas_size // 2, 0, anchor=tk.NW, image=self.img_tk)

    def get_particles(self):
        with torch.no_grad():
            if 'ddlp' in self.model_type:
                if self.seq_img is None:
                    img_seq_pil = [Image.open(img_path) for img_path in self.seq_path]
                    img_seq = torch.stack(
                        [ToTensor()(img)[None, :3].to(torch.device(self.device_name)) for img in img_seq_pil], dim=1)
                    if self.ds_name == 'phyre':
                        img_seq = 1.0 - img_seq
                else:
                    img_seq = self.seq_img
                particle_dict = self.model(img_seq, deterministic=True, forward_dyn=True)
            else:
                img_t = ToTensor()(self.img)[None, :3].to(torch.device(self.device_name))
                particle_dict = self.model(img_t, deterministic=True)
        return particle_dict

    def normalize_kp(self, kp, normalize=True):
        if normalize:
            # canvas_size -> [-1, 1]
            kp_n = kp / (self.canvas_size - 1)  # [0, 1]
            kp_n = kp_n * 2 - 1  # [-1, 1]
            kp_n = torch.tensor(kp_n, device=torch.device(self.device_name), dtype=torch.float).flip(dims=(-1,))
        else:
            # [-1, 1] -> canvas_size
            kp_n = kp * 0.5 + 0.5  # [0, 1]
            kp_n = kp_n * (self.canvas_size - 1)
            kp_n = kp_n.int()
            kp_n = kp_n.flip(dims=(-1,))
        return kp_n

    def add_keypoints(self, keypoints, scales, scale_multipliers, obj_ons, features, feature_indices, glimpses):
        for i, kp in enumerate(keypoints):
            x, y = kp
            scale = scales[i]
            keypoint = KeyPoint(self.canvas, x, y, scale, i, feature_indices[i], features=features,
                                scale_multiplier=scale_multipliers[i], obj_on=obj_ons[i], glimpse=glimpses[i],
                                gui=self)
            self.keypoints.append(keypoint)

    def add_keypoints_trajectory(self, keypoints, scales, scale_multipliers, obj_ons, features, feature_indices,
                                 glimpses):
        # assume nested lists [[t=1:3], [t=1:3], ...]
        n_kp = len(keypoints)
        for i in range(n_kp):
            timesteps = len(keypoints[i])
            tmp_kp = []
            for j in reversed(range(timesteps)):
                x, y = keypoints[i][j]
                scale = scales[i][j]
                if j == 0:
                    target_kp = (keypoints[i][timesteps - 1][0], keypoints[i][timesteps - 1][1])
                else:
                    target_kp = None
                keypoint = KeyPoint(self.canvas, x, y, scale, i, feature_indices[i][j], features=features[:, j],
                                    scale_multiplier=scale_multipliers[i][j], obj_on=obj_ons[i][j],
                                    glimpse=glimpses[i][j],
                                    to_kp=target_kp, timestep=j, gui=self)
                tmp_kp.append(keypoint)
                # self.keypoints.append(keypoint)
            self.keypoints += list(reversed(tmp_kp))

    def decode_particles(self, kp=None, scales=None, depths=None, obj_ons=None, features=None, bg=None):
        if kp is None:
            kp = self.original_keypoints
        if scales is None:
            scales = self.original_scales
        if depths is None:
            depths = self.original_depths
        if obj_ons is None:
            obj_ons = self.original_obj_ons
        if features is None:
            features = self.original_features
        if bg is None:
            bg = self.original_bg
        z_kp = self.normalize_kp(kp, normalize=True).reshape(-1, self.n_frames, 2)  # [n_kp, T, 2]
        z_kp = z_kp.permute(1, 0, 2).contiguous()  # [T, n_kp, 2]
        z_scales = torch.tensor(scales, device=torch.device(self.device_name), dtype=torch.float).reshape(-1,
                                                                                                          self.n_frames,
                                                                                                          2)
        z_scales = z_scales.permute(1, 0, 2).contiguous()  # # [T, n_kp, 2]
        z_depths = torch.tensor(depths, device=torch.device(self.device_name), dtype=torch.float).reshape(-1,
                                                                                                          self.n_frames,
                                                                                                          1)
        # [n_kp, T, 1]
        z_depths = z_depths.permute(1, 0, 2).contiguous()  # [T, n_kp, 1]
        z_obj_ons = torch.tensor(obj_ons, device=torch.device(self.device_name), dtype=torch.float).reshape(-1,
                                                                                                            self.n_frames)  # [n_kp, T]
        z_obj_ons = z_obj_ons.permute(1, 0).contiguous()  # [T, n_kp]
        z_features = torch.tensor(features,
                                  device=torch.device(self.device_name), dtype=torch.float).reshape(-1, self.n_frames,
                                                                                                    self.original_features.shape[
                                                                                                        -1])
        z_features = z_features.permute(1, 0, 2).contiguous()  # [T, n_kp, F]
        z_bg = torch.tensor(bg, device=torch.device(self.device_name), dtype=torch.float).reshape(self.n_frames,
                                                                                                  self.original_bg.shape[
                                                                                                      -1])  # [T, F]
        decoder_dict = self.model.decode_all(z_kp, z_features, z_bg, z_obj_ons, z_depth=z_depths, z_scale=z_scales)
        return decoder_dict

    def update_image(self):
        # Get updated keypoints coordinates and scales
        updated_coordinates = [kp.get_coordinates() for kp in self.keypoints]
        updated_scales = [kp.get_scale() for kp in self.keypoints]
        updated_scale_multipliers = [kp.get_scale_multiplier() for kp in self.keypoints]
        updated_features = [kp.get_features() for kp in self.keypoints]
        updated_features_indices = [kp.get_features_index() for kp in self.keypoints]
        updated_obj_ons = [kp.get_obj_on() for kp in self.keypoints]

        # Convert coordinates and scales to NumPy arrays
        updated_coordinates = np.array(updated_coordinates)  # [n_kp * n_frames, 2]
        updated_scales = np.array(updated_scales)
        updated_features = np.array(updated_features)
        updated_features_indices = np.array(updated_features_indices)
        updated_scale_multipliers = np.array(updated_scale_multipliers)
        updated_obj_ons = np.array(updated_obj_ons)

        if self.model_type != 'dlp':
            # assume (diffuse-)ddlp
            # copy features to all timesteps, modify coordinates to stay on the line
            updated_scale_multipliers = updated_scale_multipliers.reshape(-1, self.n_frames, 1)
            updated_scale_multipliers[:, 1:] = updated_scale_multipliers[:, :1]
            updated_scales = self.original_scales * (1 / updated_scale_multipliers)
            updated_scale_multipliers = updated_scale_multipliers.reshape(-1, )

            updated_obj_ons = updated_obj_ons.reshape(-1, self.n_frames)
            updated_obj_ons[:, 1:] = updated_obj_ons[:, :1]
            updated_obj_ons = updated_obj_ons.reshape(-1)

            updated_features_indices = updated_features_indices.reshape(-1, self.n_frames)
            updated_features = self.original_features[updated_features_indices[:, 0]]
            updated_features_indices = updated_features_indices.reshape(-1, )
            updated_features = updated_features.reshape(-1, updated_features.shape[-1])

            updated_coordinates = updated_coordinates.reshape(-1, self.n_frames, updated_coordinates.shape[-1])
            updated_coordinates = self.transform_coordinates(self.coordinates, updated_coordinates)
            # updated_coordinates = new_coor.reshape(-1, new_coor.shape[-1])

        print(updated_coordinates)
        # decode particles
        decoder_dict = self.decode_particles(updated_coordinates, updated_scales, self.original_depths, updated_obj_ons,
                                             updated_features, self.original_bg)
        self.img = ToPILImage()(decoder_dict['rec'][0].clamp(0, 1).cpu())

        # if 'ddlp' in self.model_type:
        #     images = decoder_dict['rec']
        #     self.seq_img = images.reshape(1, self.n_frames, *images.shape[1:]).clamp(0, 1).cpu()

        self.load_image()

        # Plot the updated keypoints
        for kp in self.keypoints:
            if kp.kp_label is not None:
                kp.kp_label.destroy()
            if kp.slider is not None:
                kp.slider.destroy()
                kp.slider_label.destroy()
            if kp.slider_obj is not None:
                kp.slider_obj.destroy()
                kp.slider_obj_label.destroy()
            if kp.scroller is not None:
                kp.scroller.destroy()
                kp.scroller_label.destroy()
            if kp.gcanvas is not None:
                kp.gcanvas.destroy()
                kp.img_tk = None

        self.keypoints = []

        # ORIGINAL
        if self.model_type == 'dlp':
            glimpses = decoder_dict['dec_objects'][0].clamp(0, 1).cpu()
            alpha, rgb = torch.split(glimpses, [1, 3], dim=1)
            rgba = alpha * rgb  # [n_particles, 3, h, w]
            rgba = rgba.clamp(0, 1).cpu()
            glimpses = [ToPILImage()(rgba[i]) for i in range(rgba.shape[0])]
            self.add_keypoints(keypoints=updated_coordinates,
                               scales=self.original_scales, scale_multipliers=updated_scale_multipliers,
                               obj_ons=updated_obj_ons,
                               features=self.original_features, feature_indices=updated_features_indices,
                               glimpses=glimpses)
        else:
            glimpses = decoder_dict['dec_objects']
            alpha, rgb = torch.split(glimpses, [1, 3], dim=2)
            rgba = alpha * rgb  # [T, n_particles, 3, h, w]
            rgba = rgba.clamp(0, 1).permute(1, 0, 2, 3, 4).cpu()  # [n_particles, T, 3, h, w]
            glimpses = []
            for i in range(rgba.shape[0]):
                kp_glimpses = [ToPILImage()(rgba[i, j]) for j in range(rgba.shape[1])]
                glimpses.append(kp_glimpses)
            self.add_keypoints_trajectory(keypoints=updated_coordinates.reshape(-1, self.n_frames,
                                                                                updated_coordinates.shape[-1]),
                                          scales=self.original_scales,
                                          scale_multipliers=updated_scale_multipliers.reshape(-1, self.n_frames),
                                          obj_ons=updated_obj_ons.reshape(-1, self.n_frames),
                                          features=self.original_features,
                                          feature_indices=updated_features_indices.reshape(-1, self.n_frames),
                                          glimpses=glimpses)
        n_kp = len(self.keypoints)
        self.coordinates = updated_coordinates
        self.scales = updated_scales
        self.features = updated_features
        self.obj_ons = updated_obj_ons

        # NEW - re-encode
        # get particles
        # particle_dict = self.get_particles()
        # if self.model_type == 'dlp':
        #     kp = particle_dict['z'][0]  # [n_kp, 2], [-1, 1]
        #     kp = self.normalize_kp(kp, normalize=False)
        #     kp = kp.cpu().numpy()
        #
        #     scales = particle_dict['z_scale'][0].cpu().numpy()
        #     depths = particle_dict['z_depth'][0].cpu().numpy()
        #     features = particle_dict['z_features'][0].cpu().numpy()
        #     obj_ons = particle_dict['obj_on'][0].cpu().numpy()
        #     bg = particle_dict['z_bg'][0].cpu().numpy()
        #
        #     glimpses = particle_dict['dec_objects_original']
        #     alpha, rgb = torch.split(glimpses, [1, 3], dim=2)
        #     rgba = alpha * rgb  # [1, n_particles, 3, h, w]
        #     rgba = rgba[0].clamp(0, 1).cpu()
        #     glimpses = [ToPILImage()(rgba[i]) for i in range(rgba.shape[0])]
        # else:
        #     # ddlp: [T, n_kp, features]
        #     kp = particle_dict['z']  # [T, n_kp, 2], [-1, 1]
        #     kp = self.normalize_kp(kp, normalize=False)
        #     kp = kp.permute(1, 0, 2).cpu().numpy()  # [n_kp, T, features]
        #
        #     scales = particle_dict['z_scale'].permute(1, 0, 2).cpu().numpy()
        #     depths = particle_dict['z_depth'].permute(1, 0, 2).cpu().numpy()
        #     features = particle_dict['z_features'].permute(1, 0, 2).cpu().numpy()
        #     obj_ons = particle_dict['obj_on'].permute(1, 0).cpu().numpy()
        #     bg = particle_dict['z_bg'].cpu().numpy()  # [T, f]
        #
        #     glimpses = particle_dict['dec_objects_original']
        #     alpha, rgb = torch.split(glimpses, [1, 3], dim=2)
        #     rgba = alpha * rgb  # [T, n_particles, 3, h, w]
        #     rgba = rgba.clamp(0, 1).permute(1, 0, 2, 3, 4).cpu()  # [n_particles, T, 3, h, w]
        #     glimpses = []
        #     for i in range(rgba.shape[0]):
        #         kp_glimpses = [ToPILImage()(rgba[i, j]) for j in range(rgba.shape[1])]
        #         glimpses.append(kp_glimpses)
        #
        # self.coordinates = kp
        # self.scales = scales
        # self.features = features
        # self.obj_ons = obj_ons
        # self.depths = self.original_depths = depths
        # self.original_bg = bg
        # # Add keypoints
        # if self.model_type == 'dlp':
        #     self.add_keypoints(keypoints=kp, scales=scales, scale_multipliers=updated_scale_multipliers,
        #                        obj_ons=obj_ons,
        #                        features=features, feature_indices=updated_features_indices,
        #                        glimpses=glimpses)
        # else:
        #     feature_indices = np.array(list(range(len(kp))))[:, None].repeat(self.n_frames, axis=1)
        #     self.add_keypoints_trajectory(keypoints=kp, scales=scales,
        #                                   scale_multipliers=self.original_scale_multiplires,
        #                                   obj_ons=obj_ons,
        #                                   features=features,
        #                                   feature_indices=feature_indices,
        #                                   glimpses=glimpses)

        if self.hide_particles.get():
            self.load_image()

    def clear_buttons(self):
        if self.update_btn is not None:
            self.update_btn.destroy()
            self.update_btn = None
        if self.reset_btn is not None:
            self.reset_btn.destroy()
            self.reset_btn = None
        if (self.model_type == 'ddlp' or self.model_type == 'diffuse_ddlp') and self.play_btn is not None:
            self.play_btn.destroy()
            self.play_btn = None
        if self.hide_kp_label is not None:
            self.hide_kp_label.destroy()
            self.hide_kp_label = None
        if self.hide_kp is not None:
            self.hide_kp.destroy()
            self.hide_kp = None

    def reset_buttons(self):
        self.clear_buttons()
        n_kp = len(self.keypoints)
        # Create buttons
        self.update_btn = ttk.Button(root, text="Update", command=self.get_update_img_func, style='my.TButton')
        # self.update_btn.pack(side=tk.BOTTOM, padx=10, pady=10, ipadx=10, ipady=10)
        self.update_btn.grid(row=5, column=1, padx=10, pady=10, ipadx=10, ipady=10)

        self.reset_btn = ttk.Button(root, text="Reset", command=self.reset_image, style='my.TButton')
        # self.reset_btn.pack(side=tk.TOP, padx=10, pady=10, ipadx=10, ipady=10)
        self.reset_btn.grid(row=3, column=0, padx=10, pady=10, ipadx=10, ipady=10)

        if 'ddlp' in self.model_type:
            self.play_btn = ttk.Button(root, text="Play", command=self.play_video, style='my.TButton')
            # self.play_btn.pack(side=tk.BOTTOM, padx=10, pady=10, ipadx=10, ipady=10)
            self.play_btn.grid(row=3, column=2, padx=10, pady=10, ipadx=10, ipady=10)

    def reset_image(self):
        # Clear the canvas and keypoints list
        for kp in self.keypoints:
            if kp.kp_label is not None:
                kp.kp_label.destroy()
            if kp.slider is not None:
                kp.slider.destroy()
                kp.slider_label.destroy()
            if kp.slider_obj is not None:
                kp.slider_obj.destroy()
                kp.slider_obj_label.destroy()
            if kp.scroller is not None:
                kp.scroller.destroy()
                kp.scroller_label.destroy()
            if kp.gcanvas is not None:
                kp.gcanvas.destroy()
                kp.img_tk = None

        # self.update_btn.destroy()
        # self.reset_btn.destroy()
        self.canvas.delete('all')
        self.canvas.destroy()
        self.keypoints = []
        self.selected_keypoints = []
        self.selection_rect = None
        self.selection_start = None
        self.reset_buttons()
        self.hide_particles = tk.BooleanVar(value=False)
        self.hide_kp_label = ttk.Label(
            root,
            text='  hide particles:  ', font=('Arial', 10), background='#818485', foreground='black'
        )
        self.hide_kp_label.grid(row=4, column=0)
        self.hide_kp = ttk.Checkbutton(root, variable=self.hide_particles, command=self.on_hide_kp)
        self.hide_kp.grid(row=5, column=0)
        self.canvas = tk.Canvas(root, width=self.canvas_size, height=self.canvas_size)
        # self.canvas.pack()
        self.canvas.grid(row=3, column=1)
        # Reload and display the original image
        self.img = None
        self.load_image()
        # Add keypoints again if needed
        if self.model_type == 'dlp':
            self.add_keypoints(keypoints=self.original_keypoints, scales=self.original_scales,
                               scale_multipliers=self.original_scale_multiplires, obj_ons=self.original_obj_ons,
                               features=self.original_features, feature_indices=self.original_feature_indices,
                               glimpses=self.original_glimpses)
        else:
            self.add_keypoints_trajectory(keypoints=self.original_keypoints, scales=self.original_scales,
                                          scale_multipliers=self.original_scale_multiplires,
                                          obj_ons=self.original_obj_ons,
                                          features=self.original_features,
                                          feature_indices=self.original_feature_indices,
                                          glimpses=self.original_glimpses)
        self.coordinates = self.original_keypoints
        self.scales = self.original_scales
        self.features = self.original_features
        self.obj_ons = self.original_obj_ons
        self.original_scale_multiplires = np.ones_like(self.obj_ons)
        self.depths = self.original_depths
        self.canvas.bind('<ButtonPress-1>', self.on_canvas_press)
        self.canvas.bind('<B1-Motion>', self.on_canvas_drag)
        self.canvas.bind('<ButtonRelease-1>', self.on_canvas_release)

    def close_all(self):
        for kp in self.keypoints:
            if kp.kp_label is not None:
                kp.kp_label.destroy()
            if kp.slider is not None:
                kp.slider.destroy()
                kp.slider_label.destroy()
            if kp.slider_obj is not None:
                kp.slider_obj.destroy()
                kp.slider_obj_label.destroy()
            if kp.scroller is not None:
                kp.scroller.destroy()
                kp.scroller_label.destroy()
            if kp.gcanvas is not None:
                kp.gcanvas.destroy()
                kp.img_tk = None

    def unroll_particles(self, num_steps=60):
        # Get current keypoints coordinates and scales
        updated_coordinates = [kp.get_coordinates() for kp in self.keypoints]
        updated_scales = [kp.get_scale() for kp in self.keypoints]
        updated_features = [kp.get_features() for kp in self.keypoints]
        updated_obj_ons = [kp.get_obj_on() for kp in self.keypoints]

        # Convert coordinates and scales to NumPy arrays
        kp = np.array(updated_coordinates)
        scales = np.array(updated_scales)
        depths = self.original_depths
        features = np.array(updated_features)
        obj_ons = np.array(updated_obj_ons)
        bg = self.original_bg

        if kp is None:
            kp = self.original_keypoints
        if scales is None:
            scales = self.original_scales
        if depths is None:
            depths = self.original_depths
        if obj_ons is None:
            obj_ons = self.original_obj_ons
        if features is None:
            features = self.original_features
        if bg is None:
            bg = self.original_bg

        z_kp = self.normalize_kp(kp, normalize=True).reshape(-1, self.n_frames, 2)  # [n_kp, T, 2]
        z_kp = z_kp.permute(1, 0, 2).unsqueeze(0).contiguous()  # [T, n_kp, 2]
        z_scales = torch.tensor(scales, device=torch.device(self.device_name), dtype=torch.float).reshape(-1,
                                                                                                          self.n_frames,
                                                                                                          2)
        z_scales = z_scales.permute(1, 0, 2).contiguous().unsqueeze(0)  # # [T, n_kp, 2]
        z_depths = torch.tensor(depths, device=torch.device(self.device_name), dtype=torch.float).reshape(-1,
                                                                                                          self.n_frames,
                                                                                                          1)
        # [n_kp, T, 1]
        z_depths = z_depths.permute(1, 0, 2).contiguous().unsqueeze(0)  # [T, n_kp, 1]
        z_obj_ons = torch.tensor(obj_ons, device=torch.device(self.device_name), dtype=torch.float).reshape(-1,
                                                                                                            self.n_frames)  # [n_kp, T]
        z_obj_ons = z_obj_ons.permute(1, 0).contiguous().unsqueeze(0)  # [T, n_kp]
        z_features = torch.tensor(features,
                                  device=torch.device(self.device_name), dtype=torch.float).reshape(-1, self.n_frames,
                                                                                                    self.original_features.shape[
                                                                                                        -1])
        z_features = z_features.permute(1, 0, 2).contiguous().unsqueeze(0)  # [T, n_kp, F]
        z_bg = torch.tensor(bg, device=torch.device(self.device_name), dtype=torch.float).reshape(self.n_frames,
                                                                                                  self.original_bg.shape[
                                                                                                      -1]).unsqueeze(0)
        z_bg = z_bg[:, :1].repeat(1, self.n_frames, 1)
        # [T, F]
        # Generate the sequence of images based on the current particles
        # dynamics
        dyn_out = self.model.dyn_module.sample(z_kp, z_scales, z_obj_ons, z_depths, z_features, z_bg,
                                               steps=num_steps, deterministic=True)
        z_dyn, z_scale_dyn, z_obj_on_dyn, z_depth_dyn, z_features_dyn, z_bg_features_dyn = dyn_out
        # decode
        z_dyn = z_dyn[:, 1:].reshape(-1, *z_dyn.shape[2:])
        z_features_dyn = z_features_dyn[:, 1:].reshape(-1, *z_features_dyn.shape[2:])
        z_bg_features_dyn = z_bg_features_dyn[:, 1:].reshape(-1, *z_bg_features_dyn.shape[2:])
        z_obj_on_dyn = z_obj_on_dyn[:, 1:].reshape(-1, *z_obj_on_dyn.shape[2:])
        z_depth_dyn = z_depth_dyn[:, 1:].reshape(-1, *z_depth_dyn.shape[2:])
        z_scale_dyn = z_scale_dyn[:, 1:].reshape(-1, *z_scale_dyn.shape[2:])

        # z_dyn = z_dyn.reshape(-1, *z_dyn.shape[2:])
        # z_features_dyn = z_features_dyn.reshape(-1, *z_features_dyn.shape[2:])
        # z_bg_features_dyn = z_bg_features_dyn.reshape(-1, *z_bg_features_dyn.shape[2:])
        # z_obj_on_dyn = z_obj_on_dyn.reshape(-1, *z_obj_on_dyn.shape[2:])
        # z_depth_dyn = z_depth_dyn.reshape(-1, *z_depth_dyn.shape[2:])
        # z_scale_dyn = z_scale_dyn.reshape(-1, *z_scale_dyn.shape[2:])
        dec_out = self.model.decode_all(z_dyn, z_features_dyn, z_bg_features_dyn, z_obj_on_dyn,
                                        z_depth=z_depth_dyn, z_scale=z_scale_dyn)
        rec_dyn = dec_out['rec'].clamp(0, 1)
        rec_dyn = rec_dyn.reshape(-1, *rec_dyn.shape[1:]).cpu()  # [T, 3, h, w]
        return rec_dyn

    def view_instructions(self):
        print(self.instructions_str)
        usage_window = tk.Toplevel(root, bg='#818485')
        usage_window.title("(D)DLP GUI Usage Guide")
        usage_label = tk.Label(usage_window, text=self.instructions_str, font=("Arial", 16), anchor="w", justify="left")

        def close_window():
            usage_window.destroy()

        usage_label.pack()
        close_button = ttk.Button(usage_window, text="Close", command=close_window, style="my.TButton")
        close_button.pack()

    def play_video(self):
        sequence_of_images = self.unroll_particles()
        sequence_of_images = [ToPILImage()(sequence_of_images[i]) for i in range(sequence_of_images.shape[0])]
        # Open a new window to display the video
        video_window = tk.Toplevel(root, bg='#818485')
        video_window.title("Generated Video")
        video_label = tk.Label(video_window)

        def update_video_frame(idx):
            if idx < len(sequence_of_images):
                image = sequence_of_images[idx].resize((self.canvas_size, self.canvas_size), Image.LANCZOS)
                image_tk = ImageTk.PhotoImage(image)
                video_label.configure(image=image_tk)
                video_label.image = image_tk
                idx += 1
            else:
                idx = 0
            video_window.after(50, update_video_frame, idx)

        def close_window():
            video_window.destroy()

        video_label.pack()
        close_button = ttk.Button(video_window, text="Close", command=close_window, style="my.TButton")
        close_button.pack()
        update_video_frame(0)

    def play_video_threaded(self):
        threading.Thread(target=self.play_video).start()

    def update_image_t(self):
        # Get updated keypoints coordinates and scales
        updated_coordinates = [kp.get_coordinates() for kp in self.keypoints]
        updated_scales = [kp.get_scale() for kp in self.keypoints]
        updated_features = [kp.get_features() for kp in self.keypoints]
        updated_obj_ons = [kp.get_obj_on() for kp in self.keypoints]

        # Convert coordinates and scales to NumPy arrays
        updated_coordinates = np.array(updated_coordinates)
        updated_scales = np.array(updated_scales)
        updated_features = np.array(updated_features)
        updated_obj_ons = np.array(updated_obj_ons)

        inter_steps = np.linspace(0, 1, num=self.num_interpolation_steps, endpoint=True)

        imgs = []

        for k in range(len(inter_steps)):
            t = inter_steps[k]
            inter_coordinates = t * np.array(updated_coordinates) + (1 - t) * self.coordinates.reshape(-1,
                                                                                                       self.coordinates.shape[
                                                                                                           -1])
            inter_scales = t * np.array(updated_scales) + (1 - t) * self.scales.reshape(-1, self.scales.shape[-1])
            inter_features = t * np.array(updated_features) + (1 - t) * self.features.reshape(-1,
                                                                                              self.features.shape[-1])
            inter_obj_ons = t * np.array(updated_obj_ons) + (1 - t) * self.obj_ons.reshape(-1, )
            decoder_dict = self.decode_particles(inter_coordinates, inter_scales, self.original_depths,
                                                 inter_obj_ons, inter_features, self.original_bg)
            imgs.append(ToPILImage()(decoder_dict['rec'][0].clamp(0, 1).cpu()))

        def func(index):
            if index < len(imgs):
                print(f'interpolation index: {index}')
                self.img = imgs[index]
                self.img_tk = ImageTk.PhotoImage(self.img.resize((self.canvas_size, self.canvas_size), Image.LANCZOS))
                # self.canvas.itemconfig(self.img_container, image=self.img_tk)
                # self.canvas.delete('all')
                self.canvas.create_image(0, 0, anchor=tk.NW, image=self.img_tk)
                index = index + 1
                self.root.after(20, func, index)

        # for l in range(len(imgs)):
        #     self.img = imgs[l]
        #     self.img_tk = ImageTk.PhotoImage(self.img.resize((self.canvas_size, self.canvas_size), Image.ANTIALIAS))
        #     # self.canvas.itemconfig(self.img_container, image=self.img_tk)
        #     self.canvas.delete('all')
        #     self.canvas.create_image(0, 0, anchor=tk.NW, image=self.img_tk)
        #     # self.load_image()
        #     self.root.after(4000, func, l)
        func(0)

        # Plot the updated keypoints
        # print(f'{updated_features_indices}')
        # if self.model_type == 'dlp':
        #     glimpses = decoder_dict['dec_objects'][0].clamp(0, 1).cpu()
        #     alpha, rgb = torch.split(glimpses, [1, 3], dim=1)
        #     rgba = alpha * rgb  # [n_particles, 3, h, w]
        #     rgba = rgba.clamp(0, 1).cpu()
        #     glimpses = [ToPILImage()(rgba[i]) for i in range(rgba.shape[0])]
        #     self.add_keypoints(keypoints=updated_coordinates,
        #                        scales=self.original_scales, scale_multipliers=updated_scale_multipliers,
        #                        obj_ons=updated_obj_ons,
        #                        features=self.original_features, feature_indices=updated_features_indices,
        #                        glimpses=glimpses)
        # else:
        #     glimpses = decoder_dict['dec_objects']
        #     alpha, rgb = torch.split(glimpses, [1, 3], dim=2)
        #     rgba = alpha * rgb  # [T, n_particles, 3, h, w]
        #     rgba = rgba.clamp(0, 1).permute(1, 0, 2, 3, 4).cpu()  # [n_particles, T, 3, h, w]
        #     glimpses = []
        #     for i in range(rgba.shape[0]):
        #         kp_glimpses = [ToPILImage()(rgba[i, j]) for j in range(rgba.shape[1])]
        #         glimpses.append(kp_glimpses)
        #     self.add_keypoints_trajectory(keypoints=updated_coordinates.reshape(-1, self.n_frames,
        #                                                                         updated_coordinates.shape[-1]),
        #                                   scales=self.original_scales,
        #                                   scale_multipliers=updated_scale_multipliers.reshape(-1, self.n_frames),
        #                                   obj_ons=updated_obj_ons.reshape(-1, self.n_frames),
        #                                   features=self.original_features,
        #                                   feature_indices=updated_features_indices.reshape(-1, self.n_frames),
        #                                   glimpses=glimpses)
        # n_kp = len(self.keypoints)
        # self.coordinates = updated_coordinates
        # self.scales = updated_scales
        # self.features = updated_features
        # self.obj_ons = updated_obj_ons

    def update_image_threaded(self):

        def check_if_ready(thread):
            if thread.is_alive():
                self.root.after(200, check_if_ready, thread)
            else:
                self.root.after(1000, self.update_image)

        trd = threading.Thread(target=self.update_image_t)
        trd.start()
        self.root.after(1, check_if_ready, trd)

    def transform_coordinates(self, original_coor, new_coor):
        # original_coor, new_coor: [n_kp, T, 2]
        # consider only first and last keypoints
        # original_coor = np.flip(original_coor, axis=-1)
        # new_coor = np.flip(new_coor, axis=-1)
        n_kp, T, _ = original_coor.shape
        p_0, p_last = original_coor[:, 0], original_coor[:, -1]  # [n_kp, 2]
        p_0_new, p_last_new = new_coor[:, 0], new_coor[:, -1]  # [n_kp, 2]

        # interpolate
        inter_steps = np.linspace(0, 1, endpoint=True, num=T)[None, :, None].repeat(repeats=n_kp,
                                                                                    axis=0)  # [n_kp, T, 1]
        updated_coor = (1 - inter_steps) * p_0_new[:, None, :] + inter_steps * p_last_new[:, None, :]
        return updated_coor

        # # Calculate translation vectors
        # translation_original = p_0
        # translation_new = p_0_new
        #
        # # Calculate the translation matrix
        # translation_matrix = np.eye(3)[None, :].repeat(repeats=p_0.shape[0], axis=0)
        # translation_matrix[:, :2, 2] = translation_new - translation_original
        #
        # # Calculate rotation and scaling parameters
        # angle_original = np.arctan2(p_last[:, 1] - p_0[:, 1], p_last[:, 0] - p_0[:, 0])
        # # angle_original = np.arctan2(p_last[:, 0] - p_0[:, 0], p_last[:, 1] - p_0[:, 1])
        # angle_new = np.arctan2(p_last_new[:, 1] - p_0_new[:, 1], p_last_new[:, 0] - p_0_new[:, 0])
        # # angle_new = np.arctan2(p_last_new[:, 0] - p_0_new[:, 0], p_last_new[:, 1] - p_0_new[:, 1])
        # scale_original = np.linalg.norm(p_last - p_0, axis=-1)
        # scale_new = np.linalg.norm(p_last_new - p_0_new, axis=-1)
        #
        # # Calculate the rotation matrix
        # rotation_matrix = np.eye(3)[None, :].repeat(repeats=p_0.shape[0], axis=0)
        # rotation_matrix[:, 0, 0] = np.cos(angle_new - angle_original)
        # rotation_matrix[:, 1, 0] = -np.sin(angle_new - angle_original)
        # rotation_matrix[:, 0, 1] = np.sin(angle_new - angle_original)
        # rotation_matrix[:, 1, 1] = np.cos(angle_new - angle_original)
        #
        # # Calculate the scaling matrix
        # scaling_matrix = np.eye(3)[None, :].repeat(repeats=p_0.shape[0], axis=0)
        # scaling_matrix[:, 0, 0] = scale_new / scale_original
        # scaling_matrix[:, 1, 1] = scale_new / scale_original
        #
        # # Combine the matrices to obtain the final transformation matrix
        # # transformation_matrix = np.matmul(scaling_matrix, np.matmul(rotation_matrix, translation_matrix))
        # transformation_matrix = np.matmul(np.matmul(translation_matrix, rotation_matrix), scaling_matrix)
        #
        # # Apply the transformation to p1 and p2
        # source_coor = original_coor.reshape(-1, original_coor.shape[-1])  # [n_kp * T, 2]
        # source_coor = np.concatenate([source_coor, np.ones(shape=(source_coor.shape[0], 1))], axis=-1)
        # updated_coor = np.matmul(transformation_matrix, source_coor.reshape(n_kp, T, -1).transpose(0, 2, 1))
        # updated_coor = updated_coor.transpose(0, 2, 1)[:, :, :2]
        # updated_coor = np.concatenate([new_coor[:, :1], updated_coor[:, 1:-1], new_coor[:, -1:]], axis=1)
        # # updated_coor = np.flip(updated_coor, axis=-1)
        # return updated_coor


def test_function():
    import matplotlib.pyplot as plt
    def transform_keypoints(original_keypoints, modified_keypoints):
        # Convert the keypoints to numpy arrays
        p0, p1, p2, p3 = np.array(original_keypoints)
        p0_new, p3_new = np.array(modified_keypoints)

        # Calculate translation vectors
        translation_original = p0
        translation_new = p0_new

        # Calculate the translation matrix
        translation_matrix = np.eye(3)
        translation_matrix[:2, 2] = translation_new - translation_original

        # Calculate rotation and scaling parameters
        angle_original = np.arctan2(p3[1] - p0[1], p3[0] - p0[0])
        angle_new = np.arctan2(p3_new[1] - p0_new[1], p3_new[0] - p0_new[0])
        scale_original = np.linalg.norm(p3 - p0)
        scale_new = np.linalg.norm(p3_new - p0_new)

        # Calculate the rotation matrix
        rotation_matrix = np.eye(3)
        rotation_matrix[:2, :2] = [[np.cos(angle_new - angle_original), -np.sin(angle_new - angle_original)],
                                   [np.sin(angle_new - angle_original), np.cos(angle_new - angle_original)]]

        # Calculate the scaling matrix
        scaling_matrix = np.eye(3)
        scaling_matrix[0, 0] = scale_new / scale_original
        scaling_matrix[1, 1] = scale_new / scale_original

        # Combine the matrices to obtain the final transformation matrix
        transformation_matrix = np.matmul(scaling_matrix, np.matmul(rotation_matrix, translation_matrix))

        # Apply the transformation to p1 and p2
        p1_transformed = np.matmul(transformation_matrix, np.append(p1, 1))[:2]
        p2_transformed = np.matmul(transformation_matrix, np.append(p2, 1))[:2]

        return p1_transformed.tolist(), p2_transformed.tolist()

    original_keypoints = [(10, 10), (25, 25), (32, 20), (40, 40)]
    modified_keypoints = [(5, 5), (40, 40)]

    p1_transformed, p2_transformed = transform_keypoints(original_keypoints, modified_keypoints)

    # Unpack the original keypoints
    p0, p1, p2, p3 = original_keypoints

    # Create a figure and axis objects
    fig, ax = plt.subplots()

    # Plot the original keypoints
    ax.plot([p0[0], p3[0]], [p0[1], p3[1]], 'r-', label='Original Trajectory')
    ax.plot(p0[0], p0[1], 'ro', label='p0')
    ax.plot(p1[0], p1[1], 'ro', label='p1')
    ax.plot(p2[0], p2[1], 'ro', label='p2')
    ax.plot(p3[0], p3[1], 'ro', label='p3')

    p0_new, p3_new = modified_keypoints
    # Plot the modified keypoints
    ax.plot([p0_new[0], p3_new[0]], [p0_new[1], p3_new[1]], 'g-', label='Modified Trajectory')
    ax.plot(p0_new[0], p0_new[1], 'go', label='p0_new')
    ax.plot(p3_new[0], p3_new[1], 'go', label='p3_new')

    # Plot the transformed keypoints
    ax.plot(p1_transformed[0], p1_transformed[1], 'bo', label='p1_transformed')
    ax.plot(p2_transformed[0], p2_transformed[1], 'bo', label='p2_transformed')

    # Set axis limits and labels
    ax.set_xlim([0, 50])
    ax.set_ylim([0, 50])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')

    # Add a legend
    ax.legend()

    # Display the plot
    plt.show()


if __name__ == '__main__':
    # test_function()

    # Create the Tkinter root window
    # root = tk.Tk()
    root = ThemedTk(theme='equilux', background=True)
    # root = ThemedTk()
    width = root.winfo_screenwidth()
    height = root.winfo_screenheight()
    root_w = max(width // 4, 512)
    root_h = max(height // 2, 512)
    print(f'setting GUI size: {root_w}x{root_h}')
    root.minsize(root_w, root_h)
    root.configure(bg='#818485')
    # root.configure(bg='#f53b47')

    # Specify the image path
    # image_path = "/media/newhd/data/obj3d/train/1/test_10.png"

    # # Create sample NumPy arrays of keypoints and scales
    # keypoints = np.array([[50, 50], [25, 80], [10, 5]])
    # scales = np.array([1.0, 1.0, 1.0])

    # Create an instance of the GUI class
    gui = GUI(root)

    # Start the Tkinter event loop
    root.mainloop()
