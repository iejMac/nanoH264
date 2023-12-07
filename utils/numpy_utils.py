import io
import pickle
import numpy as np


def serialize_array(arr):
  buffer = io.BytesIO()
  pickle.dump(arr, buffer)
  buffer.seek(0)
  return buffer.read()


def deserialize_array(byte_arr):
  buffer = io.BytesIO(byte_arr)
  buffer.seek(0)
  return pickle.load(buffer)
