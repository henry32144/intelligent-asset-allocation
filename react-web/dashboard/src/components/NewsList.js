import React from 'react';
import { makeStyles } from '@material-ui/core/styles';
import Grid from '@material-ui/core/Grid';
import NewsCard from './NewsCard';



const useStyles = makeStyles((theme) => ({
  root: {
  },
  listSubHeader: {
    textAlign: 'initial'
  },
  listTitle: {
    margin: theme.spacing(1, 1, 1),
    display: "inline-flex"
  },
  saveButton: {
    marginLeft: "auto",
    height: "36px",
    width: "64px"
  },
  emptyText: {
    margin: theme.spacing(2, 0, 2),
    textAlign: 'center'
  },
  buttonProgress: {
    position: 'absolute',
    top: '50%',
    left: '50%',
    marginTop: -12,
    marginLeft: -12,
  }
}));


function NewsList(props) {
  const classes = useStyles();

  return (
    <Grid className={classes.root} container direction="column">
      {props.newsData.map((news) => (
        <NewsCard 
          key={news.id} 
          date={news.date}
          title={news.title}
          companyName={news.companyName}
          paragraph={news.paragraph}
        >
        </NewsCard>
      ))}
    </Grid>
  );
}

export default NewsList