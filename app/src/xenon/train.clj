(ns xenon.train
  (:require [clojure.string :as string]
            [org.apache.clojure-mxnet.callback :as callback]
            [org.apache.clojure-mxnet.eval-metric :as eval-metric]
            [org.apache.clojure-mxnet.initializer :as init]
            [org.apache.clojure-mxnet.io :as mx-io]
            [org.apache.clojure-mxnet.module :as m]
            [org.apache.clojure-mxnet.ndarray :as ndarray]
            [org.apache.clojure-mxnet.optimizer :as optimizer]
            [org.apache.clojure-mxnet.symbol :as sym]
            [xenon.parameters :as x-params]
            [xenon.data-iter :as x-data]))

(defn get-model
  ([]
   (get-model x-params/models-prefix))
  ([model-prefix]
   (let [model (m/load-checkpoint {:prefix model-prefix :epoch 0})]
     {:msymbol (m/symbol model)
      :arg-params (m/arg-params model)
      :aux-params (m/aux-params model)})))

(defn get-fine-tune-model
  [{:keys [msymbol arg-params num-classes layer-name]
    :or {layer-name "flatten0"}}]
  (let [all-layers (sym/get-internals msymbol)
        net (sym/get all-layers (str layer-name "_output"))]
    {:net (as-> net data
            (sym/fully-connected "fc1" {:data data :num-hidden num-classes})
            (sym/softmax-output "softmax" {:data data}))
     :new-args (->> arg-params
                    (remove (fn [[k v]] (string/includes? k "fc1")))
                    (into {}))}))

(defn init-model
  [devs msymbol arg-params aux-params]
  (let [test-data (x-data/test-iter)]
  (-> (m/module msymbol {:contexts devs})
      (m/bind {:data-shapes (mx-io/provide-data test-data)
               :label-shapes (mx-io/provide-label test-data)})
      (m/init-params {:arg-params arg-params
                      :aux-params aux-params
                      :allow-missing true}))))
;  (.dispose test-data)))

(def fit-params
  {:intializer (init/xavier {:rand-type "gaussian"
                             :factor-type "in"
                             :magnitude 2})
   :optimizer (optimizer/adam {:learning-rate x-params/learning-rate
                               :beta1 0.9
                               :beta2 0.999})
   :batch-size x-params/batch-size
   :batch-end-callback (callback/speedometer x-params/batch-size 10)}
  )

(defn fit-epoch
  [model epoch]
  (let [train-data (x-data/train-iter)
        eval-data (x-data/val-iter)]
    (m/fit model {:train-data train-data
                  :eval-data eval-data
                  :num-epoch 1
                  :fit-params (m/fit-params fit-params)})
    (m/save-checkpoint model {:prefix x-params/saved-mod-prefix :epoch epoch :save-opt-states true})
    (.dispose train-data)
    (.dispose eval-data)
    (System/gc)))

(defn fit
  [model]
  (dorun (for [epoch (range x-params/train-epochs)] (fit-epoch model epoch))))

(defn fine-tune!
  ([]
   (fine-tune! x-params/default-context))
  ([devs]
   (let [{:keys [msymbol arg-params aux-params] :as model} (get-model)
         {:keys [net new-args]} (get-fine-tune-model (merge model {:num-classes 2}))
         model (init-model devs net new-args arg-params)]
     (fit model))))

(comment
  (fine-tune!)

)
