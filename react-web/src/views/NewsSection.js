import React from 'react';
import { makeStyles } from '@material-ui/core/styles';
import Grid from '@material-ui/core/Grid';
import NewsList from '../components/NewsList'
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

export default function NewsSection(props) {
  const classes = useStyles();

  return (
    <div className={classes.sectionRoot}>
      <Typography className={classes.sectionTitle} variant="h5">
        News
      </Typography>
      <NewsList
        newsData={props.newsData}
      >
      </NewsList>
    </div>
  );
}