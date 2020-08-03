import React from 'react';
import { makeStyles } from '@material-ui/core/styles';
import Grid from '@material-ui/core/Grid';
import Typography from '@material-ui/core/Typography';
import { BASEURL } from '../Constants';

const useStyles = makeStyles((theme) => ({
  sectionRoot: {
    margin: theme.spacing(2, 0, 2),
  },
  sectionTitle: {
    margin: theme.spacing(0, 0, 2),
  }
}));

export default function PerformanceSection(props) {
  const classes = useStyles();

  return (
    <div className={classes.sectionRoot}>
      <Typography className={classes.sectionTitle} variant="h5">
        Performance
      </Typography>
    </div>
  );
}