(ns xenon.parameters
  (:require [org.apache.clojure-mxnet.context :as context]))

;; prepare
(def orig-data-dir "MURA-v1.1/")
(def train-data-dir (str orig-data-dir "train"))
(def valid-data-dir (str orig-data-dir "valid"))
(def num-channels 3)
(def test-split 0.80)

(def image-net-mean [0.485 0.456 0.406 0.])
(def image-net-std [0.229 0.224 0.225 0.3787])

;; prepare + train
(def path-labels ["positive" "negative"])
(def labels ["abnormal" "normal"])
(def img-dim 224)

(def data-folder "data/")
(def train-folder (str data-folder "train"))
(def test-folder (str data-folder "test"))
(def val-folder (str data-folder "val"))

(def rec-folder "rec/")
(def rec-prefix (str rec-folder "mura"))

;; train
(def models-folder "models/")
(def model-name "mura-densenet-0") ;;"densenet-169")
(def models-prefix (str models-folder model-name))
(def saved-mod-prefix (str models-folder "mura-densenet-0"));; model-name))

(def learning-rate 0.0001)
(def batch-size 32)
(def epoch-size 32768)
(def train-epochs 1)

(def default-context [(context/gpu)])
