(define (problem airlift-problem)
  (:domain airlift)

  (:objects
    a_0 a_1 a_2 a_3 a_4 a_5 a_6 a_7 a_8 a_9 - plane
    c0 c1 c2 c3 c4 c5 c6 c7 c8 c9 c10 c11 c12 c13 c14 c15 c16 c17 c18 c19 c20 c21 c22 c23 c24 c25 c26 c27 c28 c29 c30 c31 c32 c33 c34 c35 c36 c37 c38 c39 - cargo
    airport1 airport2 airport3 airport4 airport5 airport6 airport7 airport8 airport9 airport10 airport11 airport12 airport13 airport14 - airport
  )

  (:init
    (at a_0 airport9)
    (free a_0)
    (at a_1 airport8)
    (free a_1)
    (at a_2 airport12)
    (free a_2)
    (at a_3 airport9)
    (free a_3)
    (at a_4 airport5)
    (free a_4)
    (at a_5 airport3)
    (free a_5)
    (at a_6 airport8)
    (free a_6)
    (at a_7 airport6)
    (free a_7)
    (at a_8 airport14)
    (free a_8)
    (at a_9 airport3)
    (free a_9)

    ;; cargo initial locations
    (cargo-at c0 airport12)
    (cargo-at c1 airport13)
    (cargo-at c2 airport14)
    (cargo-at c3 airport12)
    (cargo-at c39 airport13)

    ;; connectivity
    (connected airport1 airport12)
    (connected airport1 airport5)
    (connected airport1 airport3)
    (connected airport14 airport12)
    (connected airport14 airport1)
  )

  (:goal (and
    (cargo-at c0 airport10)
    (cargo-at c1 airport11)
    (cargo-at c2 airport10)
    (cargo-at c3 airport9)
    ...
    (cargo-at c39 airport11)
  ))
)
