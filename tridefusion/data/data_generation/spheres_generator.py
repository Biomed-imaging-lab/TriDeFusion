import typing as tp
from random import randint, uniform

import numpy as np
from image_processing import draw_mask_in_position
from spheres_processing import generate_circle

class SpheresGenerator:
    def __init__(self):
        pass

    def __init_synthetic_masks(self) -> None:
        self.__masks = []
        
        new_mask_shape = [self.__radius_int[1] * 2 - 1] * 3
        self.__new_musk_center = np.array(new_mask_shape) // 2
        
        for i in range(self.__radius_int[0], self.__radius_int[1] + 1):
            rad_x, rad_y = i, i
            rad_z = i * self.__scale[1] / self.__scale[0] + 1

            circle = generate_circle(rad_x, rad_y, rad_z, *self.__new_musk_center, new_mask_shape, 250)
            circle /= np.amax(circle)

            self.__masks.append(circle)
        return

    def init_params(
        self,
        scale:tp.Tuple[int] = [100, 22, 22],
        radius_int:tp.Tuple[int] = [1, 22],
        brightness_int:tp.Tuple[float] = [0.75, 1.0],
    ) -> None:
        self.__scale = scale
        self.__radius_int = radius_int
        self.__bright_int = brightness_int
        self.__init_synthetic_masks()
        return

    def generate_image(
        self,
        image_shape:tp.Tuple[int]=[40, 2048, 2048],
        spheres_cnt:int=1024 * 80,
    ) -> np.ndarray:
        image_canvas = np.zeros(image_shape, dtype="float32")
        
        rand_ints = np.random.uniform(self.__bright_int[0], self.__bright_int[1], spheres_cnt)
        
        rand_coords = np.concatenate((np.random.randint(0, image_shape[0], spheres_cnt).reshape(spheres_cnt, 1), 
                                      np.random.randint(0, image_shape[1], spheres_cnt).reshape(spheres_cnt, 1), 
                                      np.random.randint(0, image_shape[2], spheres_cnt).reshape(spheres_cnt, 1)), axis=1)
        
        rand_thicks = np.random.randint(self.__radius_int[0], self.__radius_int[1], spheres_cnt)

        for i in range(spheres_cnt):
            image_canvas = draw_mask_in_position(
                    image_canvas, self.__masks[rand_thicks[i]] * rand_ints[i], rand_coords[i], self.__new_musk_center
                )

        return image_canvas
