#
SUBDIRS := mls_eng_opus
#
# --------------------------------
.PHONY: all
all:
	for dir in ${SUBDIRS}; do \
		echo $$dir; \
		cd $$dir; \
		echo `pwd`; \
		bash prepare.sh; \
		cd ..; \
	done

.PHONY: clean
clean:
	for dir in ${SUBDIRS}; do \
		cd $$dir; \
		rm .done; \
		cd ..; \
	done

