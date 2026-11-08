from unittest.mock import patch
import test_span019_followup as original
from arena.span020_pipeline import orchestrate


class H66PipelineTests(original.FollowupTests):
    def setUp(self):
        def checked(args,launch):
            def verify(command,check):
                if '--mode' in command:
                    self.assertEqual(command[command.index('-m')+1],'arena.span020_infer')
                return launch(command,check=check)
            return orchestrate(args,verify)
        self.changed=patch.object(original,'orchestrate',checked)
        self.changed.start()

    def tearDown(self):
        self.changed.stop()
