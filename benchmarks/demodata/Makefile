# Generator fuer Embedding-Datensaetze in den vier Thesis-Stufen.
#
# Usage:
#   make all OUT=./out                       # alle vier Stufen
#   make S OUT=./out                         # nur Stufe S
#   make queries STUFE=S OUT=./out           # Queries + Ground-Truth fuer S
#   make all-queries OUT=./out               # Queries fuer alle Stufen
#
# Achtung: Ground-Truth bei XL ist rechenintensiv. Auf CPU mehrere Stunden,
# besser auf GPU (siehe README.md).

PYTHON ?= python3
OUT    ?= ./out
DIM    ?= 384
SEED   ?= 42

STUFEN := S M L XL

.PHONY: all $(STUFEN) clean queries all-queries help

help:
	@echo "Targets:"
	@echo "  all          -- alle Stufen erzeugen"
	@echo "  S|M|L|XL     -- einzelne Stufe erzeugen"
	@echo "  queries      -- Queries + Ground-Truth fuer eine Stufe (STUFE=X)"
	@echo "  all-queries  -- Queries fuer alle Stufen"
	@echo "  clean        -- Output-Verzeichnis loeschen"
	@echo ""
	@echo "Variablen: OUT=$(OUT)  DIM=$(DIM)  SEED=$(SEED)  PYTHON=$(PYTHON)"

all: $(STUFEN)

$(STUFEN):
	$(PYTHON) generate.py --output-dir $(OUT)/$@ --size $@ --dim $(DIM) --seed $(SEED)

queries:
ifndef STUFE
	$(error Bitte STUFE setzen, z.B. make queries STUFE=S)
endif
	$(PYTHON) gen_queries.py \
		--corpus-dir $(OUT)/$(STUFE) \
		--output-dir $(OUT)/$(STUFE)/queries \
		--dim $(DIM)

all-queries:
	@for s in $(STUFEN); do \
		$(MAKE) queries STUFE=$$s OUT=$(OUT); \
	done

clean:
	rm -rf $(OUT)
