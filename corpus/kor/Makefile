#
#SUBDIRS := AIHUB_002 AIHUB_001 AIHUB_137 ksponspeech klecspeech kconf2021speech kconf2023speech KOR_dental
SUBDIRS := AIHUB_123
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

