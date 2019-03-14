(ns xenon.data-iter
  (:require [org.apache.clojure-mxnet.io :refer [image-record-iter]]
            [xenon.parameters :as x-params]))

(defn- create-record-iter
  ([suffix & params]
   (image-record-iter (merge {:path-imgrec (str x-params/rec-prefix suffix)
                              :data-name "data"
                              :label-name "softmax_label"
                              :batch-size x-params/batch-size
                              :data-shape [x-params/num-channels
                                           x-params/img-dim
                                           x-params/img-dim]}
                             (first params)))))

(defn train-iter []
  (create-record-iter "-train.rec" {:shuffle true
                                    :rand-mirror true
                                    :max-rotate-angle 30}))

(defn val-iter []
  (create-record-iter "-val.rec"))

(defn test-iter []
  (create-record-iter "-test.rec"))
