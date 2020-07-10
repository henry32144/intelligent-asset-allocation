class Config(object):
    JOBS = [
        {
            'id': 'interval_test',
            'func': 'model_helper:interval_test',
            'args': '',
            'trigger': 'interval',
            'seconds': 5
        },
        {
            'id':'cron_test',
            'func':'model_helper:cron_test',
            'args': '',
            'trigger':'cron',
            'second':5
        }
    ]

    SCHEDULER_API_ENABLED = True