#!/usr/bin/env python3
from cpu_vision.ssd.mobilenetv1_ssd import create_mobilenetv1_ssd, create_mobilenetv1_ssd_predictor
import cv2
import numpy as np
import os
import torch


if __name__ == '__main__':
    # device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device=torch.device("cuda:0")
    device=torch.device("cpu")
    pkg_path = os.path.dirname(os.path.abspath(__file__))
    model_path =os.path.join(pkg_path, 'models/RealandSimul/mb1-ssd-Epoch-308-Loss-1.0236728725892104.pth')
    # model_path =os.path.join(pkg_path, 'models/RealandSimul/mb1-ssd-Epoch-1709-Loss-0.8591486492058199.pth') 
    label_path = os.path.join(pkg_path, 'models/RealandSimul/labels.txt')
    class_names = [name.strip() for name in open(label_path).readlines()]
    net = create_mobilenetv1_ssd(len(class_names), is_test=True, device=device)
    net.load(model_path)
    predictor = create_mobilenetv1_ssd_predictor(net, candidate_size=200, device=device)
    vid = cv2.VideoCapture(0)

    while(True):
        ret, cv_image = vid.read()
        image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        bboxs, labels, probs = predictor.predict(image, 10, 0.4)
        # Visualization
        for i in range(bboxs.size(0)):
            box = bboxs[i, :].numpy().astype(int)
            cv2.rectangle(cv_image, (box[0], box[1]), (box[2], box[3]), (255, 255, 0), 4)
            label = f"{class_names[labels[i]]}: {probs[i]:.2f}"
            cv2.putText(cv_image, label,
                        (box[0] + 20, box[1] + 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,  # font scale
                        (255, 0, 255),
                        2)  # line type
        cv2.imshow('result', cv_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    vid.release()
    cv2.destroyAllWindows()
