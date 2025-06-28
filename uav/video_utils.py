from threading import Thread
import cv2

def start_video_writer_thread(frame_queue, out_writer, exit_flag):
    def video_worker():
        while not exit_flag.is_set() or not frame_queue.empty():
            frame = frame_queue.get()
            if frame is None:
                break
            out_writer.write(frame)
            frame_queue.task_done()

    thread = Thread(target=video_worker, daemon=True)
    thread.start()
    return thread
