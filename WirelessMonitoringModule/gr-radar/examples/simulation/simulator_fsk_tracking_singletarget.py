#!/usr/bin/env python2
# -*- coding: utf-8 -*-
##################################################
# GNU Radio Python Flow Graph
# Title: Simulator Fsk Tracking Singletarget
# Generated: Sat Jan 20 15:59:43 2018
##################################################

if __name__ == '__main__':
    import ctypes
    import sys
    if sys.platform.startswith('linux'):
        try:
            x11 = ctypes.cdll.LoadLibrary('libX11.so')
            x11.XInitThreads()
        except:
            print "Warning: failed to XInitThreads()"

from PyQt4 import Qt
from gnuradio import analog
from gnuradio import blocks
from gnuradio import eng_notation
from gnuradio import filter
from gnuradio import gr
from gnuradio.eng_option import eng_option
from gnuradio.filter import firdes
from gnuradio.qtgui import Range, RangeWidget
from optparse import OptionParser
import radar
import sys
from gnuradio import qtgui


class simulator_fsk_tracking_singletarget(gr.top_block, Qt.QWidget):

    def __init__(self):
        gr.top_block.__init__(self, "Simulator Fsk Tracking Singletarget")
        Qt.QWidget.__init__(self)
        self.setWindowTitle("Simulator Fsk Tracking Singletarget")
        qtgui.util.check_set_qss()
        try:
            self.setWindowIcon(Qt.QIcon.fromTheme('gnuradio-grc'))
        except:
            pass
        self.top_scroll_layout = Qt.QVBoxLayout()
        self.setLayout(self.top_scroll_layout)
        self.top_scroll = Qt.QScrollArea()
        self.top_scroll.setFrameStyle(Qt.QFrame.NoFrame)
        self.top_scroll_layout.addWidget(self.top_scroll)
        self.top_scroll.setWidgetResizable(True)
        self.top_widget = Qt.QWidget()
        self.top_scroll.setWidget(self.top_widget)
        self.top_layout = Qt.QVBoxLayout(self.top_widget)
        self.top_grid_layout = Qt.QGridLayout()
        self.top_layout.addLayout(self.top_grid_layout)

        self.settings = Qt.QSettings("GNU Radio", "simulator_fsk_tracking_singletarget")
        self.restoreGeometry(self.settings.value("geometry").toByteArray())

        ##################################################
        # Variables
        ##################################################
        self.samp_rate = samp_rate = 5000000
        self.blocks_per_tag = blocks_per_tag = 2**17
        self.samp_per_freq = samp_per_freq = 1
        self.freq_res = freq_res = samp_rate/2/blocks_per_tag
        self.delta_freq = delta_freq = samp_rate/4
        self.center_freq = center_freq = 2.4e9
        self.velocity = velocity = 20
        self.v_res = v_res = freq_res*3e8/2/center_freq
        self.threshold = threshold = -120
        self.samp_discard = samp_discard = 0
        self.range_value = range_value = 10
        self.range_time = range_time = 10
        self.min_output_buffer = min_output_buffer = 2*(blocks_per_tag*samp_per_freq*2)
        self.decimator_fac = decimator_fac = 2**7
        self.R_max = R_max = 3e8/2/delta_freq

        ##################################################
        # Blocks
        ##################################################
        self._velocity_range = Range(-30, 30, 1, 20, 200)
        self._velocity_win = RangeWidget(self._velocity_range, self.set_velocity, "velocity", "counter_slider", float)
        self.top_layout.addWidget(self._velocity_win)
        self._threshold_range = Range(-200, 1000, 1, -120, 200)
        self._threshold_win = RangeWidget(self._threshold_range, self.set_threshold, 'Threshold Peak detector', "counter_slider", float)
        self.top_layout.addWidget(self._threshold_win)
        self._range_value_range = Range(0, 120, 1, 10, 200)
        self._range_value_win = RangeWidget(self._range_value_range, self.set_range_value, 'range', "counter_slider", float)
        self.top_layout.addWidget(self._range_value_win)
        self.rational_resampler_xxx_0_0 = filter.rational_resampler_ccc(
                interpolation=1,
                decimation=decimator_fac,
                taps=None,
                fractional_bw=None,
        )
        self.rational_resampler_xxx_0 = filter.rational_resampler_ccc(
                interpolation=1,
                decimation=decimator_fac,
                taps=None,
                fractional_bw=None,
        )
        self.radar_ts_fft_cc_0_0 = radar.ts_fft_cc(blocks_per_tag/decimator_fac,  "packet_len")
        self.radar_ts_fft_cc_0 = radar.ts_fft_cc(blocks_per_tag/decimator_fac,  "packet_len")
        self.radar_tracking_singletarget_0 = radar.tracking_singletarget(100, 1, v_res, 0.1, 0.001, 3, "particle")
        self.radar_static_target_simulator_cc_0 = radar.static_target_simulator_cc((range_value,), (velocity,), (1e16,), (0,), (0,), samp_rate, center_freq, -10, True, True, "packet_len")
        (self.radar_static_target_simulator_cc_0).set_min_output_buffer(524288)
        self.radar_split_fsk_cc_0 = radar.split_fsk_cc(samp_per_freq, samp_discard, "packet_len")
        (self.radar_split_fsk_cc_0).set_min_output_buffer(524288)
        self.radar_signal_generator_fsk_c_0 = radar.signal_generator_fsk_c(samp_rate, samp_per_freq, blocks_per_tag, -delta_freq/2, delta_freq/2, 1, "packet_len")
        (self.radar_signal_generator_fsk_c_0).set_min_output_buffer(524288)
        self.radar_qtgui_time_plot_0_0_0 = radar.qtgui_time_plot(250, 'range', (0,R_max), range_time, '')
        self.radar_qtgui_time_plot_0_0 = radar.qtgui_time_plot(250, 'range', (0,R_max), range_time, "TRACKING")
        self.radar_find_max_peak_c_0 = radar.find_max_peak_c(samp_rate/decimator_fac/2, threshold, 0, (), False, "packet_len")
        self.radar_estimator_fsk_0 = radar.estimator_fsk(center_freq, delta_freq, False)
        self.blocks_throttle_0 = blocks.throttle(gr.sizeof_gr_complex*1, samp_rate,True)
        (self.blocks_throttle_0).set_min_output_buffer(524288)
        self.blocks_tagged_stream_multiply_length_0_0 = blocks.tagged_stream_multiply_length(gr.sizeof_gr_complex*1, "packet_len", 1.0/decimator_fac)
        (self.blocks_tagged_stream_multiply_length_0_0).set_min_output_buffer(524288)
        self.blocks_tagged_stream_multiply_length_0 = blocks.tagged_stream_multiply_length(gr.sizeof_gr_complex*1, "packet_len", 1.0/decimator_fac)
        (self.blocks_tagged_stream_multiply_length_0).set_min_output_buffer(524288)
        self.blocks_multiply_conjugate_cc_1 = blocks.multiply_conjugate_cc(1)
        (self.blocks_multiply_conjugate_cc_1).set_min_output_buffer(524288)
        self.blocks_multiply_conjugate_cc_0 = blocks.multiply_conjugate_cc(1)
        (self.blocks_multiply_conjugate_cc_0).set_min_output_buffer(524288)
        self.blocks_add_xx_0 = blocks.add_vcc(1)
        (self.blocks_add_xx_0).set_min_output_buffer(524288)
        self.analog_noise_source_x_0 = analog.noise_source_c(analog.GR_GAUSSIAN, 0.5, 0)
        (self.analog_noise_source_x_0).set_min_output_buffer(524288)

        ##################################################
        # Connections
        ##################################################
        self.msg_connect((self.radar_estimator_fsk_0, 'Msg out'), (self.radar_qtgui_time_plot_0_0_0, 'Msg in'))
        self.msg_connect((self.radar_estimator_fsk_0, 'Msg out'), (self.radar_tracking_singletarget_0, 'Msg in'))
        self.msg_connect((self.radar_find_max_peak_c_0, 'Msg out'), (self.radar_estimator_fsk_0, 'Msg in'))
        self.msg_connect((self.radar_tracking_singletarget_0, 'Msg out'), (self.radar_qtgui_time_plot_0_0, 'Msg in'))
        self.connect((self.analog_noise_source_x_0, 0), (self.blocks_add_xx_0, 0))
        self.connect((self.blocks_add_xx_0, 0), (self.blocks_multiply_conjugate_cc_1, 0))
        self.connect((self.blocks_multiply_conjugate_cc_0, 0), (self.radar_find_max_peak_c_0, 0))
        self.connect((self.blocks_multiply_conjugate_cc_1, 0), (self.radar_split_fsk_cc_0, 0))
        self.connect((self.blocks_tagged_stream_multiply_length_0, 0), (self.radar_ts_fft_cc_0, 0))
        self.connect((self.blocks_tagged_stream_multiply_length_0_0, 0), (self.radar_ts_fft_cc_0_0, 0))
        self.connect((self.blocks_throttle_0, 0), (self.blocks_multiply_conjugate_cc_1, 1))
        self.connect((self.blocks_throttle_0, 0), (self.radar_static_target_simulator_cc_0, 0))
        self.connect((self.radar_signal_generator_fsk_c_0, 0), (self.blocks_throttle_0, 0))
        self.connect((self.radar_split_fsk_cc_0, 0), (self.rational_resampler_xxx_0, 0))
        self.connect((self.radar_split_fsk_cc_0, 1), (self.rational_resampler_xxx_0_0, 0))
        self.connect((self.radar_static_target_simulator_cc_0, 0), (self.blocks_add_xx_0, 1))
        self.connect((self.radar_ts_fft_cc_0, 0), (self.blocks_multiply_conjugate_cc_0, 1))
        self.connect((self.radar_ts_fft_cc_0_0, 0), (self.blocks_multiply_conjugate_cc_0, 0))
        self.connect((self.rational_resampler_xxx_0, 0), (self.blocks_tagged_stream_multiply_length_0, 0))
        self.connect((self.rational_resampler_xxx_0_0, 0), (self.blocks_tagged_stream_multiply_length_0_0, 0))

    def closeEvent(self, event):
        self.settings = Qt.QSettings("GNU Radio", "simulator_fsk_tracking_singletarget")
        self.settings.setValue("geometry", self.saveGeometry())
        event.accept()

    def get_samp_rate(self):
        return self.samp_rate

    def set_samp_rate(self, samp_rate):
        self.samp_rate = samp_rate
        self.set_delta_freq(self.samp_rate/4)
        self.radar_static_target_simulator_cc_0.setup_targets((self.range_value,), (self.velocity,), (1e16,), (0,), (0,), self.samp_rate, self.center_freq, -10, True, True)
        self.set_freq_res(self.samp_rate/2/self.blocks_per_tag)
        self.blocks_throttle_0.set_sample_rate(self.samp_rate)

    def get_blocks_per_tag(self):
        return self.blocks_per_tag

    def set_blocks_per_tag(self, blocks_per_tag):
        self.blocks_per_tag = blocks_per_tag
        self.set_min_output_buffer(2*(self.blocks_per_tag*self.samp_per_freq*2))
        self.set_freq_res(self.samp_rate/2/self.blocks_per_tag)

    def get_samp_per_freq(self):
        return self.samp_per_freq

    def set_samp_per_freq(self, samp_per_freq):
        self.samp_per_freq = samp_per_freq
        self.set_min_output_buffer(2*(self.blocks_per_tag*self.samp_per_freq*2))

    def get_freq_res(self):
        return self.freq_res

    def set_freq_res(self, freq_res):
        self.freq_res = freq_res
        self.set_v_res(self.freq_res*3e8/2/self.center_freq)

    def get_delta_freq(self):
        return self.delta_freq

    def set_delta_freq(self, delta_freq):
        self.delta_freq = delta_freq
        self.set_R_max(3e8/2/self.delta_freq)

    def get_center_freq(self):
        return self.center_freq

    def set_center_freq(self, center_freq):
        self.center_freq = center_freq
        self.set_v_res(self.freq_res*3e8/2/self.center_freq)
        self.radar_static_target_simulator_cc_0.setup_targets((self.range_value,), (self.velocity,), (1e16,), (0,), (0,), self.samp_rate, self.center_freq, -10, True, True)

    def get_velocity(self):
        return self.velocity

    def set_velocity(self, velocity):
        self.velocity = velocity
        self.radar_static_target_simulator_cc_0.setup_targets((self.range_value,), (self.velocity,), (1e16,), (0,), (0,), self.samp_rate, self.center_freq, -10, True, True)

    def get_v_res(self):
        return self.v_res

    def set_v_res(self, v_res):
        self.v_res = v_res

    def get_threshold(self):
        return self.threshold

    def set_threshold(self, threshold):
        self.threshold = threshold
        self.radar_find_max_peak_c_0.set_threshold(self.threshold)

    def get_samp_discard(self):
        return self.samp_discard

    def set_samp_discard(self, samp_discard):
        self.samp_discard = samp_discard

    def get_range_value(self):
        return self.range_value

    def set_range_value(self, range_value):
        self.range_value = range_value
        self.radar_static_target_simulator_cc_0.setup_targets((self.range_value,), (self.velocity,), (1e16,), (0,), (0,), self.samp_rate, self.center_freq, -10, True, True)

    def get_range_time(self):
        return self.range_time

    def set_range_time(self, range_time):
        self.range_time = range_time

    def get_min_output_buffer(self):
        return self.min_output_buffer

    def set_min_output_buffer(self, min_output_buffer):
        self.min_output_buffer = min_output_buffer

    def get_decimator_fac(self):
        return self.decimator_fac

    def set_decimator_fac(self, decimator_fac):
        self.decimator_fac = decimator_fac
        self.blocks_tagged_stream_multiply_length_0_0.set_scalar(1.0/self.decimator_fac)
        self.blocks_tagged_stream_multiply_length_0.set_scalar(1.0/self.decimator_fac)

    def get_R_max(self):
        return self.R_max

    def set_R_max(self, R_max):
        self.R_max = R_max


def main(top_block_cls=simulator_fsk_tracking_singletarget, options=None):

    from distutils.version import StrictVersion
    if StrictVersion(Qt.qVersion()) >= StrictVersion("4.5.0"):
        style = gr.prefs().get_string('qtgui', 'style', 'raster')
        Qt.QApplication.setGraphicsSystem(style)
    qapp = Qt.QApplication(sys.argv)

    tb = top_block_cls()
    tb.start()
    tb.show()

    def quitting():
        tb.stop()
        tb.wait()
    qapp.connect(qapp, Qt.SIGNAL("aboutToQuit()"), quitting)
    qapp.exec_()


if __name__ == '__main__':
    main()
