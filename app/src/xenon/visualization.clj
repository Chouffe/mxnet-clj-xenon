(ns xenon.visualization
  (:require [org.apache.clojure-mxnet.context :as context]
            [org.apache.clojure-mxnet.module :as m]
            [org.apache.clojure-mxnet.ndarray :as ndarray]
            [org.apache.clojure-mxnet.symbol :as sym]
            [quil.core :as q]
            [xenon.parameters :as x-params]
            [xenon.predict :as x-predict]
            [xenon.utils :as x-utils]))

(def feature-extractor
  (let [model (m/load-checkpoint {:prefix x-params/models-prefix :epoch 0})]
    (-> (sym/get-internals (m/symbol model))
        (sym/get "TBstage1_pool1_output")
        (m/module {:context [context/cpu]})
        (m/bind {:for-training false
                 :data-shapes [{:name "data"
                                :shape [1 x-params/num-channels
                                        x-params/img-dim x-params/img-dim]}]})
        (m/init-params {:arg-params (m/arg-params model)
                        :aux-params (m/aux-params model)}))))

(defn feature-extract
  [img-path]
  (-> (m/forward feature-extractor {:data [(x-predict/img-path->ndarray img-path)]})
      (m/outputs)
      (ffirst)))

(defn nd->scalar [m]
  (first (ndarray/->vec m)))

(defn ndget [m x y]
  (-> m
      (ndarray/at x)
      (ndarray/at y)
      (nd->scalar)))

(defn z-score->heat [z-score]
  (let [colors [(q/color 0 0 255)   ;; Blue
                (q/color 0 255 255) ;; Turquoise
                (q/color 0 255 0)   ;; Green
                (q/color 255 255 0) ;; Yellow
                (q/color 255 0 0)]  ;; Red
        offset  (-> (q/map-range z-score 0 1 0 3.999)
                    (max 0)
                    (min 3.999))]
    (q/lerp-color (nth colors offset)
                  (nth colors (inc offset))
                  (rem offset 1))))

(defn fill-fn [x y img]
  (-> (ndget img y x)
      (- (nd->scalar (ndarray/min img)))
      (/ (- (nd->scalar (ndarray/max img)) (nd->scalar (ndarray/min img))))
      (z-score->heat)))

(defn draw-grid [img scale fill-fn]
  (let [[x-len y-len] (ndarray/->vec (ndarray/shape img))
        width (* x-len scale)
        height (* y-len scale)
        setup (fn []
                (doseq [x (range x-len)
                        y (range y-len)]
                  (let [x-pos (* x scale)
                        y-pos (* y scale)]
                    (q/fill (fill-fn x y img))
                    (q/rect x-pos y-pos scale scale))))]
    (q/sketch :setup setup :size [width height])))

(comment
(def feature-vector (feature-extract "data/test/abnormal/00000.png"))

(def feature-matrix (ndarray/reshape feature-vector [128 28 28]))

(def feature-image (ndarray/at feature-matrix 93))

(draw-grid feature-image 20 fill-fn)

)
