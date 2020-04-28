import cv2
from kkimagemods.lib.streamer import Streamer

streamer = Streamer(0)
while streamer.is_open():
    print(f'Frame: {streamer.get_frame_count()}, fps: {streamer.get_fps()}')
    _, frame = streamer.get_frame()
    if frame is None:
        raise Exception(f'Failed to read frame.')

    print(frame)
    cv2.imshow('test', frame)
    # ESCキー押下で終了
    if cv2.waitKey(30) & 0xff == 27:
        break

streamer.close()
