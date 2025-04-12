from random import randint, uniform
import typing as tp

import numpy as np
from image_processing import draw_mask_in_position
from spheres_processing import generate_circle

COORDS_CNT = 1200

class SticksGenerator:
    def __init__(self):
        pass

    def __init_synthetic_masks(self) -> None:
        self.__masks = []
        
        new_mask_shape = [self.__radius_int[1] * 2 - 1] * 3
        self.__new_mask_center = np.array(new_mask_shape) // 2
        
        for i in range(self.__radius_int[0], self.__radius_int[1] + 1):
            rad_x, rad_y = i, i
            rad_z = i * self.__scale[1] / self.__scale[0] + 1

            circle = generate_circle(rad_x, rad_y, rad_z, *self.__new_mask_center, new_mask_shape, 250)
            circle /= np.amax(circle)

            self.__masks.append(circle)
        return


    def init_params(
        self,
        scale:tp.Tuple[int]=[100, 22, 22],

        radius_int:tp.Tuple[int]=[1, 22],
        radius_freq:tp.Tuple[float]=[2 * np.pi / COORDS_CNT, 8 * np.pi / COORDS_CNT],
        
        brightness_int:tp.Tuple[float]=[0.75, 1.0],
        brightness_freq:tp.Tuple[float]=[2 * np.pi / COORDS_CNT, 8 * np.pi / COORDS_CNT],
        
        phase:tp.Tuple[float]=[-np.pi, np.pi],
        phi_int:tp.Tuple[float]=[-np.pi, np.pi],
        omega_int=[0, 0],
    ) -> None:
        self.__scale = scale

        self.__radius_int = radius_int
        self.__radius_freq = radius_freq

        self.__bright_int = brightness_int
        self.__bright_freq = brightness_freq
        
        self.__phase = phase
        self.__phi_int = phi_int
        self.__omega_int = omega_int
        
        self.__init_synthetic_masks()
        return

    def __draw_line(
        self,
        img : np.ndarray,
        mask_int : tp.Tuple[int],
        radius_freq : float,
        bright_int : tp.Tuple[int],
        bright_freq : float,
        phase : float,
        coords_start : tp.Tuple[int],
        coords_end : tp.Tuple[int],
    ):
        # step 0 - generate masks indexes, positions and intensities
        brushes_coords = list()
        for i in range(coords_start.shape[0]):
            brushes_coords.append(
                np.linspace(coords_start[i], coords_end[i], COORDS_CNT)
            )

        brightnesses = (
            np.sin(np.arange(COORDS_CNT) * bright_freq + phase) * 0.5 + 0.5
        ) * (bright_int[1] - bright_int[0]) + bright_int[0]

        masks_indxs = np.rint(
            (np.sin(np.arange(COORDS_CNT) * radius_freq + phase) * 0.5 + 0.5)
            * (mask_int[1] - mask_int[0]) + mask_int[0]
        ).astype("uint8")

        last_coord = [-1, -1, -1]
        for brushes_coord in range(COORDS_CNT):
            coord = [
                int(brushes_coords[j][brushes_coord])
                for j in range(coords_start.shape[0])
            ]
            # TODO : Make code less shitcode!
            if (
                coord[0] != last_coord[0]
                or coord[1] != last_coord[1]
                or coord[2] != last_coord[2]
            ):
                current_int = brightnesses[i]
                current_mask_idx = masks_indxs[i]
                img = draw_mask_in_position(
                    img, self.__masks[current_mask_idx] * current_int, coord, self.__new_mask_center
                )
                last_coord = coord

        return img

    def __generate_img_with_stick(
        self,
        img_shape : tp.Tuple[int],
        start_coord : tp.Tuple[int],
        
        mask_int : tp.Tuple[int],
        radius_freq : float,
        
        bright_int : tp.Tuple[int],
        bright_freq : float,
        
        phase : float,
        phi : float,
        omega : float,
    ) -> np.ndarray:
        # step 0 - generate canvas
        img = np.zeros(img_shape, dtype="float32")

        # step 1 - generate start point and end
        tau_vec = np.array(
            [np.sin(omega), np.cos(phi) * np.cos(omega), np.sin(phi) * np.cos(omega)]
        )
        t_neg, t_pos = [], []
        for i in range(len(img_shape)):
            t1, t2 = (
                -start_coord[i] / tau_vec[i],
                (img_shape[i] - start_coord[i]) / tau_vec[i],
            )
            t_neg.append(t1 if t1 <= 0 else t2)
            t_pos.append(t2 if t1 <= 0 else t1)
        t1, t2 = np.amin(t_pos), np.amax(t_neg)
        coords = np.array([start_coord + tau_vec * t1, start_coord + tau_vec * t2])

        # step 3 - Generate line
        img = self.__draw_line(
            img,
            mask_int,
            radius_freq,
            bright_int,
            bright_freq,
            phase,
            coords[0],
            coords[1],
        )
        return img

    def __generate_one_stick_with_random_params(self, img_shape:tp.Tuple[int]) -> np.ndarray:
        # step 0 - randomize params
        masks_int = [
            randint(0, len(self.__masks) - 1),
            randint(0, len(self.__masks) - 1),
        ]
        masks_int.sort()
        
        bright_int = [
            uniform(self.__bright_int[0], self.__bright_int[1]),
            uniform(self.__bright_int[0], self.__bright_int[1]),
        ]
        bright_int.sort()
        
        bright_freq = uniform(self.__bright_freq[0], self.__bright_freq[1])
        radius_freq = uniform(self.__radius_freq[0], self.__radius_freq[1])
        phase = uniform(self.__phase[0], self.__phase[1])
        phi = uniform(self.__phi_int[0], self.__phi_int[1])
        omega = uniform(self.__omega_int[0], self.__omega_int[1])

        # генерируем координаты начала и конца
        start_coord = np.ndarray([len(img_shape)])
        for i in range(len(img_shape)):
            if i == 0:
                start_coord[i] = img_shape[i] // 2
            else:
                start_coord[i] = randint(
                    img_shape[i] // 10,
                    img_shape[i] - 1 - img_shape[i] // 10,
                )

        # step 1 - setup masks and thier centers
        new_stick = self.__generate_img_with_stick(
            img_shape,
            start_coord,
            masks_int,
            radius_freq,
            bright_int,
            bright_freq,
            phase,
            phi,
            omega,
        )
        
        return new_stick

    def generate_image(
        self,
        image_shape : tp.Tuple[int] = [40, 2048, 2048],
        sticks_cnt : int = 100
    ):
        image = np.zeros(image_shape, dtype="float32")

        for _ in range(sticks_cnt):
            new_stick = self.__generate_one_stick_with_random_params(image_shape)
            image = np.sum([image, new_stick], axis=0).astype("float32")
            
        image = (image / np.amax(image) * 255.0).astype("uint8")
        return image
