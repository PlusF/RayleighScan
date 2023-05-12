from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.floatlayout import FloatLayout
from kivy.properties import NumericProperty, StringProperty, ObjectProperty
from kivy.uix.popup import Popup
from kivy_garden.graph import Graph, LinePlot, ContourPlot
from kivy.config import Config
Config.set('graphics', 'width', '1000')
Config.set('graphics', 'height', '500')
from kivy.core.window import Window, Clock

import os
if os.name == 'nt':
    from pyAndorSDK2 import atmcd, atmcd_codes, atmcd_errors
else:
    atmcd = atmcd_codes = atmcd_errors = None
import numpy as np
import datetime
import time
import serial
import threading
from ConfigLoader import ConfigLoader
from hsc103controller import HSC103Controller
from utils import remove_cosmic_ray


class SaveDialog(FloatLayout):
    save = ObjectProperty(None)
    text_input = ObjectProperty(None)
    cancel = ObjectProperty(None)
    folder = StringProperty('')


class YesNoDialog(FloatLayout):
    message = ObjectProperty(None)
    yes = ObjectProperty(None)
    cancel = ObjectProperty(None)


class RASDriver(BoxLayout):
    current_temperature = NumericProperty(0)
    start_pos = ObjectProperty(np.zeros(3), force_dispatch=True)
    current_pos = ObjectProperty(np.zeros(3), force_dispatch=True)
    goal_pos = ObjectProperty(np.zeros(3), force_dispatch=True)
    progress_acquire_value = ObjectProperty(0)
    progress_scan_value = ObjectProperty(0)
    integration = ObjectProperty(30)
    accumulation = ObjectProperty(3)
    interval = ObjectProperty(50)
    msg = StringProperty('Please initialize the detector.')

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        Window.bind(on_request_close=self.quit)

        self.folder = './'

        self.cl = ConfigLoader('./config.json')
        if self.cl.mode == 'RELEASE':
            self.folder = self.cl.folder
            if not os.path.exists(self.folder):
                os.mkdir(self.folder)

        self.xdata = np.array([])
        self.ydata = np.array([])
        self.coord = np.array([])

        self.create_graph()

        self.ids.button_acquire.disabled = True
        self.ids.button_scan.disabled = True
        self.ids.button_save.disabled = True

        self.saved_previous = True
        self.popup_acquire = Popup(
            title="Confirmation",
            content=YesNoDialog(message='Previous data is not saved. Proceed?', yes=self._popup_yes_acquire, cancel=lambda:self.popup_acquire.dismiss()),
            size_hint=(0.4, 0.3)
        )
        self.popup_scan = Popup(
            title="Confirmation",
            content=YesNoDialog(message='Previous data is not saved. Proceed?', yes=self._popup_yes_scan, cancel=lambda:self.popup_scan.dismiss()),
            size_hint=(0.4, 0.3)
        )
        self.popup_save = Popup(
            title="Save file",
            content=SaveDialog(save=self.save, cancel=lambda:self.popup_save.dismiss(), folder=self.folder),
            size_hint=(0.9, 0.9)
        )

        self.open_ports()
        self.create_and_start_thread_position()

    def create_graph(self):
        # for spectrum
        self.graph_line = Graph(
            xlabel='Pixel number', ylabel='Counts',
            xmin=0, xmax=1023, ymin=0, ymax=1023,
            x_ticks_major=100, x_ticks_minor=2, y_ticks_major=200,
            x_grid_label=True, y_grid_label=True,
        )
        self.ids.graph_line.add_widget(self.graph_line)
        self.lineplot = LinePlot(color=[0, 1, 0, 1], line_width=1)
        self.graph_line.add_plot(self.lineplot)
        self.lineplot.points = [(i, i) for i in range(1024)]

        # for mapping
        self.graph_contour = Graph(
            xlabel='Pixel number', ylabel='Position',
            xmin=0, xmax=1023, ymin=0, ymax=9,
            x_ticks_major=100, x_ticks_minor=2, y_ticks_major=5,
            x_grid_label=True, y_grid_label=True,
        )
        self.ids.graph_contour.add_widget(self.graph_contour)
        self.contourplot = ContourPlot()
        self.graph_contour.add_plot(self.contourplot)
        self.contourplot.xrange = (0, 1023)
        self.contourplot.yrange = (0, 9)
        self.contourplot.data = np.arange(0, 10).reshape([10, 1]) * np.ones([10, 1024])
        self.contourplot.draw()

    def open_ports(self):
        if self.cl.mode == 'RELEASE':
            self.sdk = atmcd()
            self.ser = serial.Serial(self.cl.port, self.cl.baudrate, write_timeout=0)
            self.hsc = HSC103Controller(self.ser)
        elif self.cl.mode == 'DEBUG':
            self.sdk = None
            self.ser = None
            self.hsc = HSC103Controller(self.ser)
        else:
            raise ValueError('Error with config.json. mode must be DEBUG or RELEASE.')

    def create_and_start_thread_position(self):
        self.thread_pos = threading.Thread(target=self.update_position)
        self.thread_pos.daemon = True
        self.thread_pos.start()

    def create_and_start_thread_cool(self):
        self.thread_cool = threading.Thread(target=self.update_temperature)
        self.thread_cool.daemon = True
        self.thread_cool.start()

    def create_and_start_thread_acquire(self):
        self.prepare_acquisition()
        self.clear_things()
        self.num_pos = 1
        self.disable_buttons()
        self.start_progress_acquire()
        self.thread_acq = threading.Thread(target=self.acquire)
        self.thread_acq.daemon = True
        self.thread_acq.start()

    def create_and_start_thread_scan(self):
        self.num_pos = int(np.linalg.norm(self.goal_pos - self.start_pos) // self.interval + 1)
        if self.num_pos == 1:
            self.msg = 'Check the interval value again.'
            return
        self.prepare_acquisition()
        self.clear_things()
        black = np.zeros([1024, 1024])
        black[0, 0] = 1
        self.contourplot.data = black
        self.progress_scan_value = 0
        self.prepare_scan()
        self.disable_buttons()
        self.start_progress_acquire()
        self.start_progress_scan()
        self.thread_scan = threading.Thread(target=self.scan)
        self.thread_scan.daemon = True
        self.thread_scan.start()

    def start_progress_acquire(self):
        self.progress_acquire_value = -1 / self.integration / self.accumulation / 1.2
        self.clock_acquire = Clock.schedule_interval(self.update_progress_acquire, 1)

    def start_progress_scan(self):
        self.progress_scan_value = -1 / self.integration / self.accumulation / (self.num_pos + 1) / 1.2
        self.clock_scan = Clock.schedule_interval(self.update_progress_scan, 1)

    def disable_buttons(self):
        self.ids.button_acquire.disabled = True
        self.ids.button_scan.disabled = True
        self.ids.button_set_start.disabled = True
        self.ids.button_set_goal.disabled = True
        self.ids.button_go.disabled = True

    def activate_buttons(self):
        self.ids.button_acquire.disabled = False
        self.ids.button_scan.disabled = False
        self.ids.button_set_start.disabled = False
        self.ids.button_set_goal.disabled = False
        self.ids.button_go.disabled = False

    def clear_things(self):
        self.ydata = np.empty([0, self.xpixels])
        self.coord = np.empty([0, 3])
        self.lineplot.points = []
        self.progress_acquire_value = 0

    def initialize(self):
        # 初期化
        if self.cl.mode == 'RELEASE':
            if self.sdk.Initialize('') == atmcd_errors.Error_Codes.DRV_SUCCESS:
                self.ids.button_initialize.disabled = True
                self.msg = 'Successfully initialized. Now cooling...'
                self.sdk.SetTemperature(self.cl.temperature)
                self.sdk.CoolerON()
            else:
                self.msg = 'Initialization failed.'
                return
        elif self.cl.mode == 'DEBUG':
            self.ids.button_initialize.disabled = True
        self.create_and_start_thread_cool()

        self.ids.button_acquire.disabled = False
        self.ids.button_scan.disabled = False

        if self.cl.mode == 'RELEASE':
            ret, self.xpixels, ypixels = self.sdk.GetDetector()
            self.sdk.handle_return(ret)
        elif self.cl.mode == 'DEBUG':
            self.xpixels = 1024

    def update_temperature(self):
        if self.cl.mode == 'RELEASE':
            while True:
                ret, temperature = self.sdk.GetTemperature()
                if ret == atmcd_errors.Error_Codes.DRV_TEMP_STABILIZED:
                    break
                self.current_temperature = temperature
                time.sleep(self.cl.dt)
        elif self.cl.mode == 'DEBUG':
            self.current_temperature = self.cl.temperature
        self.msg = 'Cooling finished.'

    def update_graph_line(self):
        # TODO: show the spectrum accumulated
        ydata = self.ydata
        if self.cl.cosmic_ray_removal:
            ydata = remove_cosmic_ray(ydata)

        self.xdata = np.arange(0, self.xpixels, 1)
        self.graph_line.xmin = float(np.min(self.xdata))
        self.graph_line.xmax = float(np.max(self.xdata))
        self.graph_line.ymin = float(np.min(ydata[-1]))
        self.graph_line.ymax = float(max(np.max(ydata[-1]), np.min(ydata[-1]) + 0.1))
        self.lineplot.points = [(x, y) for x, y in zip(self.xdata, ydata[-1])]

    def update_graph_contour(self):
        # TODO: show the spectrum accumulated
        map_data = self.ydata
        if self.cl.cosmic_ray_removal:
            map_data = remove_cosmic_ray(map_data)

        self.xdata = np.arange(0, self.xpixels, 1)
        self.graph_contour.xmax = self.xpixels - 1
        self.graph_contour.ymax = self.num_pos * self.accumulation - 1
        self.contourplot.xrange = (0, self.xpixels - 1)
        self.contourplot.yrange = (0, len(self.ydata) - 1)

        self.contourplot.data = map_data.reshape(self.ydata.shape[::-1])

    def update_position(self):
        while True:
            self.hsc.get_position()
            msg = self.hsc.recv()

            try:
                pos_list = list(map(lambda x: int(x) * self.hsc.um_per_pulse, msg.split(',')))
            except ValueError:
                print(msg)
                time.sleep(self.cl.dt)
                continue

            self.current_pos = np.array(pos_list)
            time.sleep(self.cl.dt)

    def go(self, x, y, z):
        try:
            pos = np.array([x, y, z], float)
        except ValueError:
            self.msg = 'invalid value.'
            return

        if self.cl.mode == 'RELEASE':
            self.hsc.move_linear(pos - self.current_pos)
        elif self.cl.mode == 'DEBUG':
            pass

    def set_integration(self, val):
        try:
            integration = float(val)
        except ValueError:
            self.msg = 'Invalid value.'
            return

        if not (0.03 <= integration <= 120):  # なんとなく120秒を上限に．宇宙線の量を考えると妥当か？
            self.msg = 'Invalid value.'
            self.integration = 30
        else:
            self.msg = 'Set integration time.'
            self.integration = integration

    def set_accumulation(self, val):
        try:
            accumulation = int(val)
        except ValueError:
            self.msg = 'Invalid value.'
            return

        if accumulation < 1:
            self.msg = 'Invalid value.'
            self.accumulation = 1
        else:
            self.msg = 'Set accumulation.'
            self.accumulation = accumulation

    def set_interval(self, val):
        try:
            interval = float(val)
        except ValueError:
            self.msg = 'Invalid value.'
            return

        if interval <= 0:
            self.msg = 'Invalid value.'
            self.interval = 1
        else:
            self.msg = 'Set interval.'
            self.interval = interval

    def update_progress_acquire(self, dt):
        self.progress_acquire_value += 1 / self.integration / self.accumulation / 1.2  # prevent from exceeding
        if self.progress_acquire_value > 1:
            self.progress_acquire_value -= 1

    def update_progress_scan(self, dt):
        self.progress_scan_value += 1 / self.integration / self.accumulation / (self.num_pos + 1) / 1.2  # prevent from exceeding

    def start_acquire(self):
        if self.saved_previous:
            self.create_and_start_thread_acquire()
            return
        self.popup_acquire.open()

    def start_scan(self):
        if self.saved_previous:
            self.create_and_start_thread_scan()
            return
        self.popup_scan.open()

    def _popup_yes_acquire(self):
        self.create_and_start_thread_acquire()
        self.popup_acquire.dismiss()

    def _popup_yes_scan(self):
        self.create_and_start_thread_scan()
        self.popup_scan.dismiss()

    def prepare_acquisition(self):
        if self.cl.mode == 'RELEASE':
            self.sdk.handle_return(self.sdk.SetAcquisitionMode(atmcd_codes.Acquisition_Mode.SINGLE_SCAN))
            self.sdk.handle_return(self.sdk.SetReadMode(atmcd_codes.Read_Mode.FULL_VERTICAL_BINNING))
            self.sdk.handle_return(self.sdk.SetTriggerMode(atmcd_codes.Trigger_Mode.INTERNAL))
            self.sdk.handle_return(self.sdk.SetExposureTime(self.integration))
            self.sdk.handle_return(self.sdk.PrepareAcquisition())
        elif self.cl.mode == 'DEBUG':
            print('prepare acquisition')

    def acquire(self, during_scan=False):
        for i in range(self.accumulation):
            if self.cl.mode == 'RELEASE':
                self.sdk.handle_return(self.sdk.StartAcquisition())
                self.sdk.handle_return(self.sdk.WaitForAcquisition())
                ret, spec, first, last = self.sdk.GetImages16(1, 1, self.xpixels)
                self.ydata = np.append(self.ydata, np.array([spec]), axis=0)
                self.sdk.handle_return(ret)
            elif self.cl.mode == 'DEBUG':
                time.sleep(self.integration)
                print(f'acquired {i}')
                spec = np.expand_dims(np.sin(np.linspace(-np.pi, np.pi, self.xpixels)), axis=0) * np.random.randint(1, 10)
                noise = np.random.random(self.xpixels) * 10
                cosmic_ray = np.zeros(self.xpixels)
                cosmic_ray[np.random.randint(0, self.xpixels)] = 100
                spec += noise + cosmic_ray
                self.ydata = np.append(self.ydata, spec, axis=0)
            self.coord = np.append(self.coord, self.current_pos.reshape([1, 3]), axis=0)
            self.update_graph_line()

        if not during_scan:
            Clock.unschedule(self.clock_acquire)
            self.activate_buttons()
        self.progress_acquire_value = 1
        self.saved_previous = False
        self.ids.button_save.disabled = False

    def prepare_scan(self):
        self.hsc.set_speed_max()
        self.hsc.move_abs(self.start_pos)
        distance = np.max(self.current_pos - self.start_pos)
        time.sleep(distance / self.hsc.max_speed + 1)

    def scan(self):
        number = 0
        while number < self.num_pos:
            time_left = np.ceil((self.num_pos - number) * self.integration * self.accumulation / 60)
            self.msg = f'Acquisition {number + 1} of {self.num_pos}... {time_left} minutes left.'

            point = self.start_pos + (self.goal_pos - self.start_pos) * number / (self.num_pos - 1)
            if self.cl.mode == 'RELEASE':
                self.hsc.move_abs(point)
                distance = np.max(self.current_pos - self.start_pos)
                time.sleep(distance / self.hsc.max_speed + 1)
            elif self.cl.mode == 'DEBUG':
                self.current_pos = point

            self.acquire(during_scan=True)
            self.update_graph_contour()
            number += 1

        self.progress_acquire_value = 1
        self.progress_scan_value = 1
        Clock.unschedule(self.clock_acquire)
        Clock.unschedule(self.clock_scan)
        self.activate_buttons()
        self.saved_previous = False
        self.ids.button_save.disabled = False
        self.msg = 'Scan finished.'

    def save(self, path, filename):
        filename = os.path.basename(filename)
        if '.txt' not in filename:
            filename += '.txt'

        with open(os.path.join(path, filename), 'w') as f:
            now = datetime.datetime.now()
            f.write(f'# time: {now.strftime("%Y-%m-%d-%H-%M")}\n')
            f.write(f'# integration: {self.integration}\n')
            f.write(f'# accumulation: {self.accumulation}\n')
            f.write(f'# interval: {self.interval}\n')
            f.write(f'pos_x,{",".join(self.coord[:, 0].astype(str))}\n')
            f.write(f'pos_y,{",".join(self.coord[:, 1].astype(str))}\n')
            f.write(f'pos_z,{",".join(self.coord[:, 2].astype(str))}\n')
            for x, y in zip(self.xdata.astype(str), self.ydata.T.astype(str)):
                f.write(f'{x},{",".join(y)}\n')

        self.popup_save.dismiss()
        self.saved_previous = True
        self.popup_save.folder = path
        self.msg = f'Successfully saved to "{os.path.join(path, filename)}".'

    def quit(self, instance):
        if self.cl.mode == 'RELEASE':
            self.sdk.ShutDown()
            self.ser.close()


class RASApp(App):
    def build(self):
        self.driver = RASDriver()
        return self.driver


if __name__ == '__main__':
    RASApp().run()
