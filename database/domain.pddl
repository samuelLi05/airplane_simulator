(define (domain airlift)
  (:requirements :strips :typing)

  (:types 
    plane cargo airport
  )

  (:predicates
    ;; positions
    (at ?p - plane ?a - airport)
    (cargo-at ?c - cargo ?a - airport)

    ;; cargo load state
    (in ?c - cargo ?p - plane)

    ;; connectivity
    (connected ?a1 - airport ?a2 - airport)

    ;; capacity tracking (optional, can be extended)
    (free ?p - plane) ;; means plane has space to load more cargo
  )

  ;; ================
  ;; Actions
  ;; ================

  (:action fly
    :parameters (?p - plane ?from - airport ?to - airport)
    :precondition (and
      (at ?p ?from)
      (connected ?from ?to)
    )
    :effect (and
      (not (at ?p ?from))
      (at ?p ?to)
    )
  )

  (:action load
    :parameters (?c - cargo ?p - plane ?a - airport)
    :precondition (and
      (at ?p ?a)
      (cargo-at ?c ?a)
      (free ?p)
    )
    :effect (and
      (not (cargo-at ?c ?a))
      (in ?c ?p)
      (not (free ?p)) ;; assumes unit cargo capacity
    )
  )

  (:action unload
    :parameters (?c - cargo ?p - plane ?a - airport)
    :precondition (and
      (at ?p ?a)
      (in ?c ?p)
    )
    :effect (and
      (not (in ?c ?p))
      (cargo-at ?c ?a)
      (free ?p) ;; free space after unloading
    )
  )
)
