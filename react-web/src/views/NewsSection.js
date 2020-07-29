import React from 'react';
import { makeStyles } from '@material-ui/core/styles';
import Grid from '@material-ui/core/Grid';
import NewsCard from '../components/NewsCard'

const useStyles = makeStyles((theme) => ({
  newsSection: {
    margin: theme.spacing(2, 0, 2),
  },
}));

export default function NewsSection(props) {
  const classes = useStyles();

  return (
    <div className={classes.newsSection}>
      <NewsCard>
      </NewsCard>
    </div>
  );
}