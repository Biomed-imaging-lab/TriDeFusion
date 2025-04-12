import typing as tp
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

from matplotlib import cm
from utils.image_loader import load_image
from plot import plot_image_slices

NOISY_IMAGE_PATH = '../Denoising/validation/real_data/01/noisy_tubes.tif'
X_SCALE, Y_SCALE, Z_SCALE = 0.022, 0.022, 0.1
CHUNK_SIZE = 128
OFFSET_SIZE = 64


class ImageChunk:
    def __init__(
        self,
        img_source: np.ndarray,
        row_start: int,
        col_start: int,
        layers: int,
        chunk_size: int,
        offset_size: int,
        row_offset_begin: int,
        row_offset_end: int,
        col_offset_begin: int,
        col_offset_end: int,
    ):
        self.__chunk_data = np.zeros(
            shape=[layers, chunk_size + 2 * offset_size, chunk_size + 2 * offset_size]
        )
        self.__chunk_data[
            :, offset_size - col_offset_begin : offset_size + chunk_size + col_offset_end,
            offset_size - row_offset_begin : offset_size + chunk_size + row_offset_end
        ] = img_source[
            :, col_start - col_offset_begin : col_start + chunk_size + col_offset_end,
            row_start - row_offset_begin : row_start + chunk_size + row_offset_end
        ]

        self.row_start = row_start
        self.col_start = col_start
        self.layers = layers
        self.rows = chunk_size
        self.cols = chunk_size
        self.__offset = offset_size
        self.shape = self.__chunk_data.shape

    def set_chunk_data(self, data: np.ndarray):
        if data.shape != self.shape:
            raise ValueError("ImageChunk 'set_chunk_data' error: shapes don't match!")
        self.__chunk_data = data

    def get_data(self) -> np.ndarray:
        return self.__chunk_data

    def get_chunk_without_offset(self) -> np.ndarray:
        return self.__chunk_data[
            :, self.__offset : -self.__offset, self.__offset : -self.__offset
        ]


class BigImageManager:
    def __init__(
        self, img: np.ndarray, chunk_size_border: int, offset_size: int, layers_count: int
    ):
        self.__img = img
        self.__img_layers, self.__img_cols, self.__img_rows = img.shape
        self.__chunk_size_border = chunk_size_border
        self.__offset_size = offset_size
        self.__layers_count = layers_count

        # Inflate image to ensure whole chunks
        self.__inflated_layers, self.__inflated_cols, self.__inflated_rows = (
            self.__img_layers,
            self.__img_cols,
            self.__img_rows,
        )
        if self.__img_rows % chunk_size_border != 0:
            self.__inflated_rows += chunk_size_border - (self.__img_rows % chunk_size_border)
        if self.__img_cols % chunk_size_border != 0:
            self.__inflated_cols += chunk_size_border - (self.__img_cols % chunk_size_border)
        if self.__img_layers % layers_count != 0:
            self.__inflated_layers += layers_count - (self.__img_layers % layers_count)

        self.__inflated_img = np.zeros(
            (self.__inflated_layers, self.__inflated_cols, self.__inflated_rows)
        )
        self.__inflated_img[
            : self.__img_layers, : self.__img_cols, : self.__img_rows
        ] = self.__img

    def split_in_chunks(self) -> tp.List[ImageChunk]:
        if self.__chunk_size_border >= self.__inflated_rows or self.__chunk_size_border >= self.__inflated_cols:
            return [
                ImageChunk(
                    self.__inflated_img,
                    0,
                    0,
                    self.__inflated_layers,
                    self.__inflated_rows,
                    self.__offset_size,
                    0,
                    0,
                    0,
                    0,
                )
            ]

        chunks_list = []
        for row_start in range(0, self.__inflated_rows, self.__chunk_size_border):
            for col_start in range(0, self.__inflated_cols, self.__chunk_size_border):
                chunk_rows = min(self.__chunk_size_border, self.__inflated_rows - row_start)
                chunk_cols = min(self.__chunk_size_border, self.__inflated_cols - col_start)

                row_offset_begin = self.__offset_size if row_start >= self.__offset_size else row_start
                row_offset_end = self.__offset_size if row_start + chunk_rows < self.__inflated_rows else 0
                col_offset_begin = self.__offset_size if col_start >= self.__offset_size else col_start
                col_offset_end = self.__offset_size if col_start + chunk_cols < self.__inflated_cols else 0

                chunks_list.append(
                    ImageChunk(
                        self.__inflated_img,
                        row_start,
                        col_start,
                        self.__inflated_layers,
                        chunk_rows,
                        self.__offset_size,
                        row_offset_begin,
                        row_offset_end,
                        col_offset_begin,
                        col_offset_end,
                    )
                )
        return chunks_list

    def async_split_in_chunks(self) -> tp.List[ImageChunk]:
        with ThreadPoolExecutor() as executor:
            chunks_list = self.split_in_chunks()
            if len(chunks_list) == 1 and chunks_list[0].rows == self.__img_rows and chunks_list[0].cols == self.__img_cols:
                return chunks_list
            futures = [executor.submit(lambda chunk=chunk: chunk) for chunk in chunks_list]
            return [future.result() for future in as_completed(futures)]

    def concatenate_chunks_into_image(self, chunks_list: tp.List[ImageChunk]) -> np.ndarray:
        new_img_inflate = np.zeros(
            (self.__inflated_layers, self.__inflated_cols, self.__inflated_rows)
        )
        for chunk in chunks_list:
            chunk_data = chunk.get_chunk_without_offset()
            new_img_inflate[
                :,
                chunk.col_start : chunk.col_start + chunk.cols,
                chunk.row_start : chunk.row_start + chunk.rows,
            ] = chunk_data
        return new_img_inflate[: self.__img_layers, : self.__img_cols, : self.__img_rows]

    def async_concatenate_chunks_into_image(self, chunks_list: tp.List[ImageChunk]) -> np.ndarray:
        with ThreadPoolExecutor() as executor:
            chunk_data_without_offset = [
                (chunk, chunk.get_chunk_without_offset()) for chunk in chunks_list
            ]
        return self.concatenate_chunks_into_image([chunk for chunk, _ in chunk_data_without_offset])



if __name__ == "__main__":
    img = load_image(NOISY_IMAGE_PATH)
    img_manager = BigImageManager(img, chunk_size_border=CHUNK_SIZE, offset_size=OFFSET_SIZE, layers_count=1)
    chunk_list = img_manager.async_split_in_chunks()
    img_merged = img_manager.async_concatenate_chunks_into_image(chunk_list)
    plot_image_slices(img_merged, cm.jet, X_SCALE, Z_SCALE, np.array(img_merged.shape) // 2)