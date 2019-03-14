(ns xenon.utils
  (:require [clojure.core.matrix :as matrix]))

(defn argmax
  "Index of maximum value in vector."
  [a]
  (first (apply max-key second (map-indexed vector a))))

(defn mean
  "Returns the average of the array elements."
  [a]
  (matrix/div (reduce matrix/add a) (count a)))

(defn std
  [a]
  (matrix/sqrt (matrix/div (matrix/square (matrix/sub a (mean a))) (count a))))
