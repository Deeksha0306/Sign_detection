import cv2 as cv
import streamlit as st
import easyocr as es
import os
import matplotlib.pyplot as plt
import numpy as np

img_path = '../images/text.png'
result = es.Reader(['en'])
reader = result.readtext(img_path)
for _, text, confidence in reader:
    print(f"{text} ({round(confidence * 100, 2)}%)")