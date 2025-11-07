const router = require('express').Router();
import analytics from '../analytics';

router.get('/summary', (req, res, next) => {
  analytics.fetchSummary()
    .then((summary) => {
      res.json(summary);
    })
    .catch(next);
});

export default router;
