# --- Tk 8.7 compatibility shim: map Variable.trace('w'/'r'/'u') -> trace_add('write'/'read'/'unset') ---
# import tkinter as _tk
from tkinter import Variable as _Variable
from y_finance_impl import compute_recommendation as y_compute_recommendation
from polygon_io_impl import compute_recommendation as p_compute_recommendation

if hasattr(_Variable, "trace_add"):
    def _legacy_trace(self, mode, callback):
        mode_map = {"w": "write", "r": "read", "u": "unset"}
        return self.trace_add(mode_map.get(mode, mode), callback)
    _Variable.trace = _legacy_trace

import FreeSimpleGUI as sg
import threading

def main_gui():
    main_layout = [
        [sg.Text("Enter Stock Symbol:"), sg.Input(key="stock", size=(20, 1), focus=True)],
        [sg.Button("Submit", bind_return_key=True), sg.Button("Exit")],
        [sg.Text("", key="recommendation", size=(50, 1))]
    ]
    
    window = sg.Window("Earnings Position Checker", main_layout)
    
    while True:
        event, values = window.read()
        if event in (sg.WINDOW_CLOSED, "Exit"):
            break

        if event == "Submit":
            window["recommendation"].update("")
            stock = values.get("stock", "")

            loading_layout = [[sg.Text("Loading...", key="loading", justification="center")]]
            loading_window = sg.Window("Loading", loading_layout, modal=True, finalize=True, size=(275, 200))

            result_holder = {}

            def worker():
                try:
                    result = p_compute_recommendation(stock)
                    result_holder['result'] = result
                except Exception as e:
                    result_holder['error'] = str(e)

            thread = threading.Thread(target=worker, daemon=True)
            thread.start()

            while thread.is_alive():
                event_load, _ = loading_window.read(timeout=100)
                if event_load == sg.WINDOW_CLOSED:
                    break
            thread.join(timeout=1)

            if 'error' in result_holder:
                loading_window.close()
                window["recommendation"].update(f"Error: {result_holder['error']}")
            elif 'result' in result_holder:
                loading_window.close()
                result = result_holder['result']

                avg_volume_bool    = result['avg_volume']
                iv30_rv30_bool     = result['iv30_rv30']
                ts_slope_bool      = result['ts_slope_0_45']
                expected_move      = result['expected_move']
                
                if avg_volume_bool and iv30_rv30_bool and ts_slope_bool:
                    title = "Recommended"
                    title_color = "#006600"
                elif ts_slope_bool and ((avg_volume_bool and not iv30_rv30_bool) or (iv30_rv30_bool and not avg_volume_bool)):
                    title = "Consider"
                    title_color = "#ff9900"
                else:
                    title = "Avoid"
                    title_color = "#800000"
                
                result_layout = [
                    [sg.Text(title, text_color=title_color, font=("Helvetica", 16))],
                    [sg.Text(f"avg_volume: {'PASS' if avg_volume_bool else 'FAIL'}", text_color="#006600" if avg_volume_bool else "#800000")],
                    [sg.Text(f"iv30_rv30: {'PASS' if iv30_rv30_bool else 'FAIL'}", text_color="#006600" if iv30_rv30_bool else "#800000")],
                    [sg.Text(f"ts_slope_0_45: {'PASS' if ts_slope_bool else 'FAIL'}", text_color="#006600" if ts_slope_bool else "#800000")],
                    [sg.Text(f"Expected Move: {expected_move}", text_color="blue")],
                    [sg.Button("OK")]
                ]
                
                result_window = sg.Window("Recommendation", result_layout, modal=True, finalize=True, size=(275, 200))
                while True:
                    event_result, _ = result_window.read(timeout=100)
                    if event_result in (sg.WINDOW_CLOSED, "OK"):
                        break
                result_window.close()
    
    window.close()

def gui():
    main_gui()

if __name__ == "__main__":
    gui()