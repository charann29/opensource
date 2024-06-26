import cv2
import numpy as np


class BirdsEyeView:
    def __init__(self, source, target, frame_width, frame_height, target_width, target_height):
        self.source = source
        self.target = target
        source = self.source.astype(np.float32)
        target = self.target.astype(np.float32)
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.scale_canvas = 6
        self.canvas_origin_x = 3400
        self.canvas_origin_y = 100
        self.canvas_width = self.scale_canvas * target_width
        self.canvas_height = self.scale_canvas * target_height
        self.divide_canvas_width_in_parts = 7
        self.matrix = cv2.getPerspectiveTransform(source, target)
        self.x = None
        self.y = None

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        reshaped_points = points.reshape(-1, 1, 2).astype(np.float32)
        transformed_points = cv2.perspectiveTransform(reshaped_points, self.matrix)
        return transformed_points.reshape(-1, 2)

    def draw_canvas_boundary(self, frame):
        cv2.line(frame, (self.canvas_origin_x, self.canvas_origin_y),
                 (self.canvas_origin_x, self.canvas_origin_y + self.canvas_height),
                 (0, 0, 0), 5)
        cv2.line(frame, (self.canvas_origin_x, self.canvas_origin_y),
                 (self.canvas_origin_x + self.canvas_width, self.canvas_origin_y),
                 (0, 0, 0), 5)
        cv2.line(frame, (self.canvas_origin_x + self.canvas_width, self.canvas_origin_y),
                 (self.canvas_origin_x + self.canvas_width, self.canvas_origin_y + self.canvas_height),
                 (0, 0, 0), 5)
        cv2.line(frame, (self.canvas_origin_x, self.canvas_origin_y + self.canvas_height),
                 (self.canvas_origin_x + self.canvas_width, self.canvas_origin_y + self.canvas_height),
                 (0, 0, 0), 5)
        return frame

    def draw_background_rectangle(self, frame):
        shapes = np.zeros_like(frame, np.uint8)
        # Draw the rectangle
        cv2.rectangle(shapes, (self.canvas_origin_x, self.canvas_origin_y),
                      (self.canvas_origin_x + self.canvas_width, self.canvas_origin_y + self.canvas_height),
                      (255, 255, 255), cv2.FILLED)
        out = frame.copy()
        alpha = 0.65
        mask = shapes.astype(bool)
        out[mask] = cv2.addWeighted(frame, alpha, shapes, 1 - alpha, 0)[mask]

        return out

    def draw_road_lines(self, frame):

        road_width = int(self.canvas_width / self.divide_canvas_width_in_parts)

        for i in range(1, self.divide_canvas_width_in_parts):
            line_p1 = (self.canvas_origin_x + i * road_width, self.canvas_origin_y + 5)
            line_p2 = (self.canvas_origin_x + i * road_width, self.canvas_origin_y + self.canvas_height - 5)

            cv2.line(frame, line_p1, line_p2, (225, 255, 255), 3)

        return frame

    def draw_road(self, frame):
        frame = self.draw_canvas_boundary(frame)
        frame = self.draw_road_lines(frame)
        frame = self.draw_background_rectangle(frame)

        return frame

    def draw_car_point(self, point, frame,  track_id, color_id):
        self.x = int(point[0] * self.scale_canvas + self.canvas_origin_x)
        self.y = int(point[1] * self.scale_canvas + self.canvas_origin_y)

        strip_width = self.canvas_width / self.divide_canvas_width_in_parts
        strip_index = int(self.x / strip_width)
        self.x = int((strip_index + 0.5) * strip_width) + 20

        cv2.circle(frame, (self.x, self.y), 8, color_id, -1)
        cv2.putText(frame, f" {track_id}", (self.x, self.y - 10), 4, cv2.FONT_HERSHEY_PLAIN, color_id, 3)
        return frame

    @staticmethod
    def speed_calculator(y_current, y_half_sec_ago, time_diff):
        distance = abs(y_half_sec_ago - y_current)   # in meters
        speed_m_s = distance / time_diff
        speed_km_h = round(speed_m_s * 3.6, 3)

        return speed_km_h

    def speed_calculation(self, prev_y_dict, source_fps):
        for divisor in range(2, source_fps):
            try:
                speed_km_h = self.speed_calculator(prev_y_dict[-1],
                                                   prev_y_dict[-int((source_fps / divisor))],
                                                   (1 / divisor))
                if source_fps / divisor < 0:
                    break
                return int(speed_km_h)
            except:
                continue

    def label_speed_on_canvas(self, speed, frame, color):
        cv2.putText(frame, f"{speed}", (self.x + 15, self.y + 20), 4, cv2.FONT_HERSHEY_PLAIN, color, 3)
