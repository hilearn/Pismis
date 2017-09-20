hash= $(shell ./bin/md5_hash.py requirements.txt)
venv = .venv/${hash}


default: update_venv

.PHONY: default

${venv}: requirements.txt
	python3 -m virtualenv --no-site-packages --python=python3.5 ${venv}
	. ${venv}/bin/activate; pip install -r requirements.txt

update_venv: requirements.txt ${venv}
	@rm -f .venv/current
	@ln -s $(hash) .venv/current
	@echo Success, to activate the development environment, run:
	@echo "\tsource .venv/current/bin/activate"
