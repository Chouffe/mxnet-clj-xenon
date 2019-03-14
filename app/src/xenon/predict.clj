(ns xenon.predict
  (:require [clojure.core.matrix :as matrix]
            [clojure.java.io :as io]
            [clojure.string :as string]
            [org.apache.clojure-mxnet.context :as context]
            [org.apache.clojure-mxnet.module :as m]
            [org.apache.clojure-mxnet.ndarray :as ndarray]
            [org.apache.clojure-mxnet.symbol :as sym]
            [mikera.image.core :as img]
            [think.image.pixel :as pixel]
            [xenon.parameters :as x-params]
            [xenon.utils :as x-utils]))

(def ensemble-count 1)

(defn extract-model
  [model]
  {:msymbol (m/symbol model)
   :arg-params (m/arg-params model)
   :aux-params (m/aux-params model)})

(defn init-model
  [{:keys [msymbol arg-params aux-params]}]
  (-> (m/module msymbol {:context x-params/default-context})
      (m/bind {:for-training false
               :data-shapes [{:name "data"
                              :shape [1 x-params/num-channels
                                      x-params/img-dim x-params/img-dim]}]})
      (m/init-params {:arg-params arg-params
                      :aux-params aux-params})))

(def ensemble
  (map #(as-> % $
          (str x-params/models-folder "mura-densenet-0");;-" $)
          (m/load-checkpoint {:prefix $ :epoch 0})
          (extract-model $)
          (init-model $))
       (range ensemble-count)))

ensemble

;; prediction

(defn img-path->ndarray
  [img-path]
  (as-> img-path $
    (img/load-image $)
    (img/resize $ x-params/img-dim x-params/img-dim)
    (img/get-pixels $)
    (reduce (fn [result pixel]
              (let [[rs gs bs] result
                    [r g b _] (pixel/unpack-pixel pixel)]
                ;;[(conj rs r) (conj gs g) (conj bs b)]))
                [(conj rs (- r 123.68)) (conj gs (- g 116.779)) (conj bs (- b 103.939))]))
            [[] [] []] $)
    (flatten $)
    (ndarray/array $ [1 x-params/num-channels x-params/img-dim x-params/img-dim])))


(defn model-predict-imgs
  [model nd-images]
  (x-utils/mean (map #(-> (m/forward model {:data [%]})
                          m/outputs
                          ffirst
                          ndarray/->vec)
                     nd-images)))

(defn predict-from-paths
  [img-paths]
  (as-> img-paths $
    (map img-path->ndarray $)
    (map #(model-predict-imgs % $) ensemble)
    (x-utils/mean $)))

(defn predict-from-dir
  [dir]
  (->> dir
       io/file
       file-seq
       (filter #(.isFile %))
       (mapv str)
       predict-from-paths))

(defn correct-pred-dir?
  [dir]
  (= (string/includes? dir "positive")
     (= 0 (x-utils/argmax (predict-from-dir dir)))))

(defn- dir->study-dirs
  [dir]
  (->> dir
       io/file
       file-seq
       (filter (fn [f] (and (.isDirectory f)
                            (or (string/includes? f "positive")
                                (string/includes? f "negative")))))
       (mapv str)))

(defn calc-acc
  [dir]
  (let [dirs (dir->study-dirs dir)]
    (as-> dirs $
      (map correct-pred-dir? $)
      (filter identity $) 
     (count $)
      (/ $ (count dirs)))))

(defn calc-kappa
  [dir]
  (let [rand-prob (/ 1.0 (count x-params/labels))]
    (/ (- (calc-acc dir) rand-prob) (- 1 rand-prob))))

(comment
  (predict-from-paths ["data/test/abnormal/00000.png"])

  (predict-from-paths ["data/test/abnormal/00022.png"])
  
  (time (predict-from-dir "MURA-v1.1/valid/XR_WRIST/patient11213/study1_positive"))

  (time (predict-from-dir "MURA-v1.1/valid/XR_WRIST/patient11213/study2_negative"))

  (time (predict-from-dir "MURA-v1.1/valid/XR_WRIST/patient11199/study1_positive"))

  (time (predict-from-dir "MURA-v1.1/valid/XR_WRIST/patient11199/study2_negative"))

  (time (calc-acc "MURA-v1.1/valid/XR_WRIST/patient11199"))

  (time (calc-kappa "data/sample-test"))

)
