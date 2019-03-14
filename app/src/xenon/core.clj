(ns xenon.core
  (:require [clojure.tools.cli :refer [cli]]
            [xenon.train]))

(defn -main [& args]
  (xenon.train/fine-tune!))
